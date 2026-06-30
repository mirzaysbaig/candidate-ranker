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

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OfflineCandidateIndexer:
    """
    Enterprise ETL Pipeline: Extracts raw candidate JSONL, separates metadata for Pandas,
    and generates dense embeddings for FAISS.
    
    All tunable constants (degree rankings, tier ordering, batch sizes, default
    fallback values, etc.) are read from the ``indexer_config`` dict, which is
    sourced from the ``indexer:`` section of config.yaml.
    """
    
    def __init__(self, input_path: str, output_dir: str, model_name: str,
            parquet_filename: str, faiss_filename: str,
            indexer_config: Dict[str, Any] | None = None):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.parquet_filename = parquet_filename
        self.faiss_filename = faiss_filename
        
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
        
        # Ensure the processed output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage arrays
        self.metadata_list: List[Dict[str, Any]] = []
        self.text_corpus: List[str] = []
        
        # Initialize the Sentence Transformer model (Runs on CPU/GPU automatically)
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.vector_dim = self.model.get_embedding_dimension()

    def _extract_candidate_features(self, candidate: dict) -> None:
        """
        Parses a single JSON candidate into structured metadata and a semantic text block.
        
        Supports two schemas:
          1. Real schema (helper_candidates.jsonl / candidate_schema.json):
             Nested structure with profile, career_history, education, skills (list of dicts),
             certifications, languages, and redrob_signals.
          2. Legacy flat schema (candidates.jsonl):
             Flat dict with id, name, skills (list of strings), experience_years.
        
        JSON Tree — Value Fetching Map (real schema):
        ─────────────────────────────────────────────
        candidate_id ─────────────────────────────→ meta["candidate_id"]
        profile
        ├── anonymized_name ──────────────────────→ meta["name"], semantic text
        ├── headline ─────────────────────────────→ semantic text
        ├── summary ──────────────────────────────→ semantic text
        ├── location ─────────────────────────────→ meta["location"], semantic text
        ├── country ──────────────────────────────→ meta["country"]
        ├── years_of_experience ──────────────────→ meta["years_of_experience"], semantic text
        ├── current_title ────────────────────────→ meta["current_title"], semantic text
        ├── current_company ──────────────────────→ meta["current_company"], semantic text
        ├── current_company_size ─────────────────→ meta["current_company_size"]
        └── current_industry ─────────────────────→ meta["current_industry"], semantic text
        career_history[] (array of objects)
        ├── [i].company ──────────────────────────→ semantic text
        ├── [i].title ────────────────────────────→ semantic text
        ├── [i].duration_months ──────────────────→ semantic text
        ├── [i].industry ─────────────────────────→ semantic text
        └── [i].description ──────────────────────→ semantic text
        education[] (array of objects)
        ├── [i].institution ──────────────────────→ semantic text
        ├── [i].degree ───────────────────────────→ meta["highest_degree"], semantic text
        ├── [i].field_of_study ───────────────────→ semantic text
        └── [i].tier ─────────────────────────────→ meta["education_tier"]
        skills[] (array of objects)
        ├── [i].name ─────────────────────────────→ semantic text
        ├── [i].proficiency ──────────────────────→ semantic text
        └── [i].duration_months ──────────────────→ semantic text
        certifications[] (array of objects)
        ├── [i].name ─────────────────────────────→ meta["certifications_count"], semantic text
        └── [i].issuer ───────────────────────────→ semantic text
        languages[] (array of objects)
        └── [i].language + proficiency ───────────→ semantic text
        redrob_signals
        ├── profile_completeness_score ───────────→ meta["profile_completeness_score"]
        ├── signup_date ──────────────────────────→ meta["signup_date"]
        ├── last_active_date ─────────────────────→ meta["last_active_date"]
        ├── open_to_work_flag ────────────────────→ meta["open_to_work_flag"]
        ├── profile_views_received_30d ───────────→ meta["profile_views_received_30d"]
        ├── applications_submitted_30d ───────────→ meta["applications_submitted_30d"]
        ├── recruiter_response_rate ──────────────→ meta["recruiter_response_rate"]
        ├── avg_response_time_hours ──────────────→ meta["avg_response_time_hours"]
        ├── skill_assessment_scores (dict) ───────→ meta["avg_assessment_score"]
        │   └── {skill_name: score, ...}              (averaged across all assessed skills)
        ├── connection_count ─────────────────────→ meta["connection_count"]
        ├── endorsements_received ────────────────→ meta["endorsements_received"]
        ├── notice_period_days ───────────────────→ meta["notice_period_days"]
        ├── expected_salary_range_inr_lpa
        │   ├── min ──────────────────────────────→ meta["expected_salary_min_lpa"]
        │   └── max ──────────────────────────────→ meta["expected_salary_max_lpa"]
        ├── preferred_work_mode ──────────────────→ meta["preferred_work_mode"]
        ├── willing_to_relocate ──────────────────→ meta["willing_to_relocate"]
        ├── github_activity_score ────────────────→ meta["github_activity_score"]
        ├── search_appearance_30d ────────────────→ meta["search_appearance_30d"]
        ├── saved_by_recruiters_30d ──────────────→ meta["saved_by_recruiters_30d"]
        ├── interview_completion_rate ────────────→ meta["interview_completion_rate"]
        ├── offer_acceptance_rate ────────────────→ meta["offer_acceptance_rate"]
        ├── verified_email ───────────────────────→ meta["verified_email"]
        ├── verified_phone ───────────────────────→ meta["verified_phone"]
        └── linkedin_connected ───────────────────→ meta["linkedin_connected"]
        """
        # ── Resolve top-level containers ──────────────────────────────────
        cand_id = candidate.get("candidate_id") or candidate.get("id") or "UNKNOWN"
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})

        # Name: real schema uses profile.anonymized_name; legacy uses root name
        name = profile.get("anonymized_name") or candidate.get("name") or ""

        # Experience: real schema nests inside profile; legacy uses root
        years_of_exp = profile.get("years_of_experience")
        if years_of_exp is None:
            years_of_exp = candidate.get("experience_years", 0.0)

        # ── Education pre-processing ──────────────────────────────────────
        education_list = candidate.get("education", [])
        # Determine highest degree and best institution tier for metadata
        highest_degree = ""
        best_tier = "unknown"
        for edu in education_list:
            if isinstance(edu, dict):
                deg = edu.get("degree", "")
                if self.degree_rank.get(deg, -1) > self.degree_rank.get(highest_degree, -1):
                    highest_degree = deg
                tier = edu.get("tier", "unknown")
                if self.tier_order.get(tier, 4) < self.tier_order.get(best_tier, 4):
                    best_tier = tier

        # ── Certifications pre-processing ─────────────────────────────────
        certifications = candidate.get("certifications", [])

        # ── Skill assessment average ──────────────────────────────────────
        assessment_scores = signals.get("skill_assessment_scores", {})
        if assessment_scores and isinstance(assessment_scores, dict):
            score_values = [v for v in assessment_scores.values() if isinstance(v, (int, float))]
            avg_assessment = round(sum(score_values) / len(score_values), 2) if score_values else self.default_missing_score
        else:
            avg_assessment = self.default_missing_score

        # ── Salary range unpacking ────────────────────────────────────────
        salary_range = signals.get("expected_salary_range_inr_lpa", {})
        salary_min = salary_range.get("min", self.default_missing_score) if isinstance(salary_range, dict) else self.default_missing_score
        salary_max = salary_range.get("max", self.default_missing_score) if isinstance(salary_range, dict) else self.default_missing_score

        # ══════════════════════════════════════════════════════════════════
        # 1. METADATA EXTRACTION  (→ Parquet → Pandas → Scorer / Ranker)
        # ══════════════════════════════════════════════════════════════════
        meta = {
            # ── Identity ──
            "id":                           cand_id,
            "candidate_id":                 cand_id,
            "name":                         name,
            # ── Profile numerics & categoricals ──
            "years_of_experience":          float(years_of_exp),
            "current_title":                profile.get("current_title", ""),
            "current_company":              profile.get("current_company", ""),
            "current_company_size":         profile.get("current_company_size", ""),
            "current_industry":             profile.get("current_industry", ""),
            "location":                     profile.get("location", ""),
            "country":                      profile.get("country", ""),
            # ── Education summary ──
            "highest_degree":               highest_degree,
            "education_tier":               best_tier,
            # ── Certifications count ──
            "certifications_count":         len(certifications),
            # ── Redrob platform signals (full extraction) ──
            "profile_completeness_score":   signals.get("profile_completeness_score", 0.0),
            "signup_date":                  signals.get("signup_date", self.default_fallback_date),
            "last_active_date":             signals.get("last_active_date", self.default_fallback_date),
            "open_to_work_flag":            signals.get("open_to_work_flag", False),
            "profile_views_received_30d":   signals.get("profile_views_received_30d", 0),
            "applications_submitted_30d":   signals.get("applications_submitted_30d", 0),
            "recruiter_response_rate":      signals.get("recruiter_response_rate", 0.0),
            "avg_response_time_hours":      signals.get("avg_response_time_hours", 0.0),
            "avg_assessment_score":         avg_assessment,
            "connection_count":             signals.get("connection_count", 0),
            "endorsements_received":        signals.get("endorsements_received", 0),
            "notice_period_days":           signals.get("notice_period_days", self.default_notice_period),
            "expected_salary_min_lpa":      salary_min,
            "expected_salary_max_lpa":      salary_max,
            "preferred_work_mode":          signals.get("preferred_work_mode", ""),
            "willing_to_relocate":          signals.get("willing_to_relocate", False),
            "github_activity_score":        signals.get("github_activity_score", self.default_missing_score),
            "search_appearance_30d":        signals.get("search_appearance_30d", 0),
            "saved_by_recruiters_30d":      signals.get("saved_by_recruiters_30d", 0),
            "interview_completion_rate":    signals.get("interview_completion_rate", 0.0),
            "offer_acceptance_rate":        signals.get("offer_acceptance_rate", self.default_missing_score),
            "verified_email":               signals.get("verified_email", False),
            "verified_phone":               signals.get("verified_phone", False),
            "linkedin_connected":           signals.get("linkedin_connected", False),
        }
        self.metadata_list.append(meta)

        # ══════════════════════════════════════════════════════════════════
        # 2. SEMANTIC TEXT EXTRACTION  (→ Embedding model → FAISS index)
        # ══════════════════════════════════════════════════════════════════
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")

        # ── Skills: include proficiency & duration for richer embeddings ──
        skills_raw = candidate.get("skills", [])
        skills_parts = []
        for skill in skills_raw:
            if isinstance(skill, dict):
                s_name = skill.get("name", "")
                s_prof = skill.get("proficiency", "")
                s_dur = skill.get("duration_months", 0)
                if s_name:
                    skill_str = s_name
                    if s_prof:
                        skill_str += f" ({s_prof}"
                        if s_dur:
                            skill_str += f", {s_dur}mo"
                        skill_str += ")"
                    skills_parts.append(skill_str)
            elif isinstance(skill, str):
                skills_parts.append(skill)
        skills_text = ", ".join(skills_parts)

        # ── Career history: include description for semantic depth ────────
        history_parts = []
        for job in candidate.get("career_history", []):
            if isinstance(job, dict):
                job_title = job.get("title", "")
                company = job.get("company", "")
                industry = job.get("industry", "")
                duration = job.get("duration_months", 0)
                description = job.get("description", "")
                entry = f"{job_title} at {company}"
                if industry:
                    entry += f" ({industry})"
                if duration:
                    entry += f" for {duration} months"
                if description:
                    entry += f": {description}"
                history_parts.append(entry)
        history_text = " | ".join(history_parts)

        # ── Education: degree + field + institution ───────────────────────
        edu_parts = []
        for edu in education_list:
            if isinstance(edu, dict):
                degree = edu.get("degree", "")
                field = edu.get("field_of_study", "")
                institution = edu.get("institution", "")
                entry = f"{degree} in {field}" if field else degree
                if institution:
                    entry += f" from {institution}"
                edu_parts.append(entry)
        education_text = "; ".join(edu_parts)

        # ── Certifications ────────────────────────────────────────────────
        cert_parts = []
        for cert in certifications:
            if isinstance(cert, dict):
                c_name = cert.get("name", "")
                c_issuer = cert.get("issuer", "")
                if c_name:
                    cert_parts.append(f"{c_name} ({c_issuer})" if c_issuer else c_name)
        certifications_text = ", ".join(cert_parts)

        # ── Languages ─────────────────────────────────────────────────────
        lang_parts = []
        for lang in candidate.get("languages", []):
            if isinstance(lang, dict):
                l_name = lang.get("language", "")
                l_prof = lang.get("proficiency", "")
                if l_name:
                    lang_parts.append(f"{l_name} ({l_prof})" if l_prof else l_name)
        languages_text = ", ".join(lang_parts)

        # ── Compile the dense "Resume String" ─────────────────────────────
        parts = []
        if name:
            parts.append(f"Candidate: {name}")
        if headline:
            parts.append(f"Headline: {headline}")
        current_title = profile.get("current_title", "")
        current_company = profile.get("current_company", "")
        if current_title and current_company:
            parts.append(f"Currently: {current_title} at {current_company}")
        current_industry = profile.get("current_industry", "")
        if current_industry:
            parts.append(f"Industry: {current_industry}")
        location = profile.get("location", "")
        country = profile.get("country", "")
        if location or country:
            parts.append(f"Location: {location}, {country}" if location and country else f"Location: {location or country}")
        if summary:
            parts.append(f"Summary: {summary}")
        if skills_text:
            parts.append(f"Skills: {skills_text}")
        if years_of_exp:
            parts.append(f"Total experience: {years_of_exp} years")
        if history_text:
            parts.append(f"Career: {history_text}")
        if education_text:
            parts.append(f"Education: {education_text}")
        if certifications_text:
            parts.append(f"Certifications: {certifications_text}")
        if languages_text:
            parts.append(f"Languages: {languages_text}")

        full_text = ". ".join(parts) + "." if parts else "Empty profile."
        self.text_corpus.append(full_text)

    def process_data(self):
        """Reads the JSONL file (supports both raw and gzip) iteratively to save RAM."""
        logger.info(f"Starting ingestion from: {self.input_path}")
        
        # Check if file is gzipped or plain text based on extension
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
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
        
        # Optimize categorical columns to reduce memory footprint
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        output_path = self.output_dir / self.parquet_filename
        logger.info(f"Saving Parquet artifact to: {output_path}")
        df.to_parquet(output_path, engine='pyarrow', index=False)

    def build_vector_index(self):
        """Passes the corpus through the embedding model and compiles the FAISS index."""
        logger.info(f"Generating dense vector embeddings (batch_size={self.embedding_batch_size}). This will take some time...")
        embeddings = self.model.encode(self.text_corpus, batch_size=self.embedding_batch_size, show_progress_bar=True)
        
        # FAISS strictly requires float32 arrays
        vector_array = np.array(embeddings).astype('float32')
        
        logger.info("Applying L2 Normalization for Cosine Similarity...")
        faiss.normalize_L2(vector_array)
        
        logger.info("Compiling FAISS IndexFlatIP...")
        index = faiss.IndexFlatIP(self.vector_dim)
        index.add(vector_array)
        
        output_path = self.output_dir / self.faiss_filename
        logger.info(f"Saving FAISS artifact to: {output_path}")
        faiss.write_index(index, str(output_path))

    def run(self):
        """Executes the pipeline sequentially."""
        self.process_data()
        self.build_metadata_parquet()
        self.build_vector_index()
        logger.info("✅ Phase 1 Complete. Offline artifacts successfully generated.")

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Strictly loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Load all paths and model config strictly from config.yaml
    config = load_config()
    
    input_file = config['paths']['raw_candidates']
    parquet_path = Path(config['paths']['parquet_path'])
    faiss_path = Path(config['paths']['index_path'])
    model_name = config['model']['name']
    indexer_config = config.get('indexer', {})
    
    # Derive output directory and filenames from the config paths
    output_directory = str(parquet_path.parent)
    
    indexer = OfflineCandidateIndexer(
        input_path=input_file,
        output_dir=output_directory,
        model_name=model_name,
        parquet_filename=parquet_path.name,
        faiss_filename=faiss_path.name,
        indexer_config=indexer_config,
    )
    
    indexer.run()