import json
import gzip
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from rank_bm25 import BM25Okapi


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OfflineCandidateIndexer:
    """
    Enterprise ETL Pipeline (Multi-Vector Retrieval):
    Extracts raw candidate JSONL, separates metadata for Pandas,
    and generates THREE distinct dense embeddings (Identity, Capability, Experience) for FAISS.
    """
    
    def __init__(self, input_path: str, output_dir: str, model_name: str,
            parquet_filename: str, 
            profile_filename: str, skills_filename: str, career_filename: str,
            indexer_config: Dict[str, Any] | None = None,
            filters_path: str = 'configs/filters.yaml'):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.parquet_filename = parquet_filename
        self.profile_filename = profile_filename
        self.skills_filename = skills_filename
        self.career_filename = career_filename
        
        # Determine bm25 filename from config if available, else default
        cfg_paths = {}
        if indexer_config:
            pass # indexer_config doesn't have the paths, they are passed as kwargs
        
        self.bm25_corpus: List[List[str]] = []
        self.bm25_filename = "bm25_index.pkl"
        
        # ── Indexer-specific config (sourced from config.yaml → indexer:) ──
        cfg = indexer_config or {}
        self.embedding_batch_size: int = cfg.get("embedding_batch_size", 32)
        self.progress_log_interval: int = cfg.get("progress_log_interval", 10000)
        self.categorical_columns: List[str] = cfg.get("categorical_columns", [
            "current_company_size", "current_industry", "country",
            "highest_degree", "education_tier", "preferred_work_mode",
        ])
        self.degree_rank: Dict[str, int] = cfg.get("degree_rank", {
            "Ph.D": 4, "M.Tech": 3, "M.E.": 3, "M.Sc": 3, "MBA": 3,
            "B.Tech": 2, "B.E.": 2, "B.Sc": 2, "BCA": 1, "Diploma": 0,
        })
        self.tier_order: Dict[str, int] = cfg.get("tier_order", {
            "tier_1": 0, "tier_2": 1, "tier_3": 2, "tier_4": 3, "unknown": 4,
        })
        defaults = cfg.get("defaults", {})
        self.default_notice_period: int = defaults.get("notice_period_days", 90)
        self.default_fallback_date: str = defaults.get("fallback_date", "1970-01-01")
        self.default_missing_score: float = defaults.get("missing_score", -1.0)
        
        # ── Load dynamic filters from filters.yaml ──
        self.filters_path = filters_path
        with open(filters_path, 'r', encoding='utf-8') as f:
            self.filters = yaml.safe_load(f)
        logger.info(f"Loaded dynamic filters from: {filters_path}")
        
        # Ensure the processed output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage arrays
        self.metadata_list: List[Dict[str, Any]] = []
        
        # THREE distinct corpora for Multi-FAISS
        self.profile_corpus: List[str] = []
        self.skills_corpus: List[str] = []
        self.career_corpus: List[str] = []
        
        # Initialize the Sentence Transformer model
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.vector_dim = self.model.get_embedding_dimension()

    def _extract_candidate_features(self, candidate: dict) -> None:
        candidate_id = candidate.get("candidate_id") or candidate.get("id", "UNKNOWN_ID")
        profile = candidate.get("profile", {})
        if not profile and "name" in candidate: 
            profile = candidate
            
        name = profile.get("anonymized_name") or profile.get("name", "Unknown Candidate")
        
        # Safe extraction for legacy vs real schema
        raw_skills = candidate.get("skills", [])
        if all(isinstance(s, str) for s in raw_skills):
            all_skills = raw_skills
        else:
            all_skills = [s.get("name", "") for s in raw_skills if isinstance(s, dict)]
            
        career = candidate.get("career_history", [])
        education = candidate.get("education", [])
        certifications = candidate.get("certifications", [])
        signals = candidate.get("redrob_signals", {})

        # ======================================================================
        # 1. PARSE METADATA (DETERMINISTIC FILTERS & SCORING)
        # ======================================================================
        
        # Map education tier (lower is better in our tier_order map)
        edu_tier = profile.get("education_tier", "unknown")
        
        # Map highest degree to a numeric rank
        highest_deg = profile.get("highest_degree", "")
        degree_score = self.degree_rank.get(highest_deg, 0)
        
        # Determine current tier of company based on size (crude heuristic)
        company_size = profile.get("current_company_size", "unknown")

        past_companies = []
        past_industries = []
        all_job_titles = []
        total_duration_months = 0
        job_durations = []
        
        for job in career:
            if isinstance(job, dict):
                comp = job.get('company', '')
                title = job.get('title', '')
                ind = job.get('industry', '')
                dur = job.get('duration_months', 0)
                
                if comp: past_companies.append(comp.lower())
                if title: all_job_titles.append(title.lower())
                if ind: past_industries.append(ind.lower())
                
                if dur > 0:
                    total_duration_months += dur
                    job_durations.append(dur)
                    
        years_of_exp = profile.get("years_of_experience", 0)
        if years_of_exp == 0 and total_duration_months > 0:
            years_of_exp = round(total_duration_months / 12, 1)
            
        avg_job_duration = 0
        if job_durations:
            avg_job_duration = sum(job_durations) / len(job_durations)

        # ── Trust / Boolean Signals ──
        verified_email = signals.get("verified_email", False)
        verified_phone = signals.get("verified_phone", False)
        linkedin_conn = signals.get("linkedin_connected", False)
        open_to_work = profile.get("open_to_work_flag", False)
        
        # ── Honeypot Detection (Tech keywords but non-tech career) ──
        honeypot_flags = 0
        
        role_config = self.filters.get('role_classifications', {})
        non_tech_titles = role_config.get('non_tech_titles', [])
        current_title = profile.get("current_title", "")
        
        non_tech = False
        if current_title:
            title_lower = current_title.lower()
            if any(nt in title_lower for nt in non_tech_titles):
                non_tech = True
        
        adv_count = sum(1 for s in raw_skills if isinstance(s, dict) and s.get("proficiency") == "advanced")
        if non_tech and adv_count >= 5:
            honeypot_flags += 10
        


        # Build Pandas Row
        meta = {
            # ── Core Identity ──
            "candidate_id":                 candidate_id,
            "name":                         name,
            
            # ── Logistics & Availability ──
            "last_active_date":             signals.get("last_active_date", self.default_fallback_date),
            "notice_period_days":           profile.get("notice_period_days", self.default_notice_period),
            "open_to_work_flag":            open_to_work,
            
            # ── Market Demand & Activity ──
            "search_appearance_30d":        signals.get("search_appearance_30d", 0),
            "saved_by_recruiters_30d":      signals.get("saved_by_recruiters_30d", 0),
            "recruiter_response_rate":      signals.get("recruiter_response_rate", self.default_missing_score),
            
            # ── Competency & Quality ──
            "github_activity_score":        signals.get("github_activity_score", self.default_missing_score),
            "avg_assessment_score":         signals.get("avg_assessment_score", self.default_missing_score),
            "profile_completeness_score":   profile.get("profile_completeness_score", 0),
            
            # ── Education & Certifications ──
            "highest_degree":               highest_deg,
            "degree_score":                 degree_score,
            "education_tier":               edu_tier,
            "certifications_count":         len(certifications),
            
            # ── Reliability & Trust ──
            "interview_completion_rate":    signals.get("interview_completion_rate", self.default_missing_score),
            "offer_acceptance_rate":        signals.get("offer_acceptance_rate", self.default_missing_score),
            "verified_email":               verified_email,
            "verified_phone":               verified_phone,
            "linkedin_connected":           linkedin_conn,
            "endorsements_received":        signals.get("endorsements_received", 0),
            "connection_count":             signals.get("connection_count", 0),
            
            # ── Extracted Job History Stats ──
            "years_of_experience":          years_of_exp,
            "avg_job_duration_months":      avg_job_duration,
            "current_company_size":         company_size,
            "current_industry":             profile.get("current_industry", "unknown"),
            "country":                      profile.get("country", "unknown"),
            
            # ── Filter Lists (for fast JD filtering in Pandas) ──
            "honeypot_flags":               honeypot_flags,
            "past_companies":               ",".join(past_companies),
            "past_industries":              ",".join(past_industries),
            "all_job_titles":               ",".join(all_job_titles),
            "all_skills":                   ",".join(all_skills),
        }
        self.metadata_list.append(meta)

        # ======================================================================
        # 2. MULTI-VECTOR CORPUS GENERATION
        # ======================================================================
        
        # ── Corpus 1: The "Identity" Vector ──
        headline = profile.get("headline", "")
        current_title = profile.get("current_title", "")
        recent_titles = [job.get("title", "") for job in career[:3] if isinstance(job, dict) and job.get("title")]
        
        profile_text = f"Headline: {headline}. Current Role: {current_title}."
        if recent_titles:
            profile_text += f" Recent Roles: {', '.join(recent_titles)}."
        self.profile_corpus.append(profile_text.strip() or "Empty profile")

        # ── Corpus 2: The "Capability" Vector ──
        # Top 20 skills to prevent infinite keyword stuffing dilution
        skills_list = all_skills[:20]
        
        cert_parts = []
        for cert in certifications:
            if isinstance(cert, dict):
                c_name = cert.get("name", "")
                c_issuer = cert.get("issuer", "")
                if c_name:
                    cert_parts.append(f"{c_name} ({c_issuer})" if c_issuer else c_name)
        
        edu_parts = []
        for edu in education:
            if isinstance(edu, dict):
                degree = edu.get("degree", "")
                field = edu.get("field_of_study", "")
                if degree and field:
                    edu_parts.append(f"{degree} in {field}")
                elif degree:
                    edu_parts.append(degree)
                    
        skills_text = f"Top Skills: {', '.join(skills_list)}."
        if cert_parts:
            skills_text += f" Certifications: {', '.join(cert_parts)}."
        if edu_parts:
            skills_text += f" Education: {', '.join(edu_parts)}."
        self.skills_corpus.append(skills_text.strip() or "No skills listed")

        # ── Corpus 3: The "Experience" Vector ──
        # We explicitly EXCLUDE the profile summary here so candidates cannot
        # inflate their experience score just by writing "Interested in AI".
        job_texts = []
        for job in career:
            if isinstance(job, dict):
                title = job.get("title", "")
                desc = job.get("description", "")
                if title or desc:
                    job_texts.append(f"{title} - {desc}".strip(" -"))
        
        career_text = ""
        if job_texts:
            career_text = f"Experience Details: {' | '.join(job_texts)}."
        self.career_corpus.append(career_text.strip() or "No experience listed")

        # ══════════════════════════════════════════════════════════════════
        # 5. BM25 SPARSE CORPUS (The Keyword Search)
        # ══════════════════════════════════════════════════════════════════
        # BM25 needs the text split into a list of lowercase words (tokens)
        full_text_for_keywords = f"{profile_text} {skills_text} {career_text}".lower()
        tokens = full_text_for_keywords.split() # Simple, lightning-fast tokenization
        self.bm25_corpus.append(tokens)

    def process_data(self):
        """Reads the JSONL file iteratively to save RAM."""
        logger.info(f"Starting ingestion from: {self.input_path}")
        
        is_gzipped = self.input_path.suffix == '.gz'
        open_func = gzip.open if is_gzipped else open
        
        with open_func(self.input_path, 'rt', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    candidate_dict = json.loads(line)
                    self._extract_candidate_features(candidate_dict)
                    
                    if (line_idx + 1) % self.progress_log_interval == 0:
                        logger.info(f"Parsed {line_idx + 1} candidates...")
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted JSON on line {line_idx}. Skipping.")
                    
        logger.info(f"Ingestion complete. {len(self.metadata_list)} valid candidates extracted.")

    def build_metadata_parquet(self):
        """Compiles the metadata list into a highly optimized Parquet file."""
        logger.info("Converting metadata to Pandas DataFrame...")
        df = pd.DataFrame(self.metadata_list)
        
        # Optimize date parsing for faster runtime calculations
        df['last_active_date'] = pd.to_datetime(df['last_active_date'], errors='coerce')
        
        # Optimize categorical columns to reduce memory footprint
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        output_path = self.output_dir / self.parquet_filename
        logger.info(f"Saving Parquet artifact to: {output_path}")
        df.to_parquet(output_path, engine='pyarrow', index=False)

    def _embed_and_save(self, corpus: List[str], filename: str):
        """Helper method to embed a corpus and save its FAISS index."""
        logger.info(f"Embedding {filename} (batch_size={self.embedding_batch_size})...")
        embeddings = self.model.encode(corpus, batch_size=self.embedding_batch_size, show_progress_bar=True)
        
        # FAISS strictly requires float32 arrays
        vector_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(vector_array)
        
        index = faiss.IndexFlatIP(self.vector_dim)
        index.add(vector_array)
        
        output_path = self.output_dir / filename
        logger.info(f"Saving FAISS artifact to: {output_path}")
        faiss.write_index(index, str(output_path))

    def build_vector_indices(self):
        """Passes the three corpora through the embedding model and compiles FAISS indices."""
        self._embed_and_save(self.profile_corpus, self.profile_filename)
        self._embed_and_save(self.skills_corpus, self.skills_filename)
        self._embed_and_save(self.career_corpus, self.career_filename)

    def build_bm25_index(self):
        """Compiles the BM25Okapi model and saves it to a pickle file."""
        logger.info("Building BM25 index from corpus...")
        bm25 = BM25Okapi(self.bm25_corpus)
        
        output_path = self.output_dir / self.bm25_filename
        logger.info(f"Saving BM25 artifact to: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(bm25, f)

    def run(self):
        """Executes the pipeline sequentially."""
        self.process_data()
        self.build_metadata_parquet()
        self.build_vector_indices()
        self.build_bm25_index()
        logger.info("✅ Phase 1 Complete. Offline FAISS and BM25 artifacts successfully generated.")

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Strictly loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Load all paths and model config strictly from config.yaml
    config = load_config()
    
    input_file = config['paths']['raw_candidates']
    parquet_path = Path(config['paths']['parquet_path'])
    profile_path = Path(config['paths']['profile_index_path'])
    skills_path = Path(config['paths']['skills_index_path'])
    career_path = Path(config['paths']['career_index_path'])
    
    model_name = config['model']['name']
    indexer_config = config.get('indexer', {})
    filters_path = config.get('filters_path', 'configs/filters.yaml')
    
    bm25_path = Path(config['paths'].get('bm25_index_path', 'data/processed/bm25_index.pkl'))
    
    # Derive output directory and filenames from the config paths
    output_directory = str(parquet_path.parent)
    
    indexer = OfflineCandidateIndexer(
        input_path=input_file,
        output_dir=output_directory,
        model_name=model_name,
        parquet_filename=parquet_path.name,
        profile_filename=profile_path.name,
        skills_filename=skills_path.name,
        career_filename=career_path.name,
        indexer_config=indexer_config,
        filters_path=filters_path,
    )
    indexer.bm25_filename = bm25_path.name
    
    indexer.run()