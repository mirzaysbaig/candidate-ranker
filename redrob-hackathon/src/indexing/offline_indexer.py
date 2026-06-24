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
    """
    
    def __init__(self, input_path: str, output_dir: str, model_name: str,
            parquet_filename: str, faiss_filename: str):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.parquet_filename = parquet_filename
        self.faiss_filename = faiss_filename
        
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
        """Parses a single JSON candidate into structured metadata and a semantic text block."""
        # Retrieve candidate ID and name supporting both flat and nested schemas
        cand_id = candidate.get("candidate_id") or candidate.get("id") or "UNKNOWN"
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        
        name = candidate.get("name") or profile.get("name") or ""
        
        # Experience: look for profile value first, then root experience_years
        years_of_exp = profile.get("years_of_experience")
        if years_of_exp is None:
            years_of_exp = candidate.get("experience_years", 0.0)
        
        # 1. METADATA EXTRACTION (For our Pandas Math later)
        meta = {
            "id": cand_id,  # Add for compatibility with ranker
            "candidate_id": cand_id,
            "name": name,   # Add for compatibility with ranker
            "years_of_experience": float(years_of_exp),
            "current_company": profile.get("current_company", ""),
            "last_active_date": signals.get("last_active_date", "1970-01-01"),
            "open_to_work_flag": signals.get("open_to_work_flag", False),
            "notice_period_days": signals.get("notice_period_days", 90),
            "recruiter_response_rate": signals.get("recruiter_response_rate", 0.0),
            "github_activity_score": signals.get("github_activity_score", -1.0)
        }
        self.metadata_list.append(meta)
        
        # 2. SEMANTIC TEXT EXTRACTION (For our AI Embeddings)
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        
        # Safely parse skills array (handles list of strings and list of dicts)
        skills_raw = candidate.get("skills", [])
        skills_list = []
        for skill in skills_raw:
            if isinstance(skill, dict):
                skills_list.append(skill.get("name", ""))
            elif isinstance(skill, str):
                skills_list.append(skill)
        skills_text = ", ".join(skills_list)
        
        # Safely parse career history
        history_list = []
        for job in candidate.get("career_history", []):
            if isinstance(job, dict):
                job_title = job.get("title", "")
                company = job.get("company", "")
                history_list.append(f"{job_title} at {company}")
        history_text = " | ".join(history_list)
        
        # Compile the dense "Resume String" dynamically
        parts = []
        if name:
            parts.append(f"Candidate name: {name}")
        if headline:
            parts.append(f"Headline: {headline}")
        if summary:
            parts.append(f"Summary: {summary}")
        if skills_text:
            parts.append(f"Skills: {skills_text}")
        if years_of_exp:
            parts.append(f"Experience: {years_of_exp} years")
        if history_text:
            parts.append(f"History: {history_text}")
        
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
                    
                    if (line_idx + 1) % 10000 == 0:
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
        
        output_path = self.output_dir / self.parquet_filename
        logger.info(f"Saving Parquet artifact to: {output_path}")
        df.to_parquet(output_path, engine='pyarrow', index=False)

    def build_vector_index(self):
        """Passes the corpus through the embedding model and compiles the FAISS index."""
        logger.info("Generating dense vector embeddings. This will take some time...")
        # batch_size=32 is safe for standard CPUs without causing memory spikes
        embeddings = self.model.encode(self.text_corpus, batch_size=32, show_progress_bar=True)
        
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
    
    # Derive output directory and filenames from the config paths
    output_directory = str(parquet_path.parent)
    
    indexer = OfflineCandidateIndexer(
        input_path=input_file,
        output_dir=output_directory,
        model_name=model_name,
        parquet_filename=parquet_path.name,
        faiss_filename=faiss_path.name
    )
    
    indexer.run()