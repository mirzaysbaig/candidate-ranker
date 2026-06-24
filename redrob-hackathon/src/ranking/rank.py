import argparse
import logging
import sys
from pathlib import Path
import yaml

# Import our custom enterprise microservices
from src.embeddings.embedding_service import get_jd_text, generate_query_vector
from src.retrieval.faiss_search import search_candidates
from src.models.scorer import apply_behavioral_math 
from src.utils.helpers import generate_reasoning, format_submission_df
from src.indexing.offline_indexer import OfflineCandidateIndexer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ranker(candidates_path: str, output_path: str, config_path: str = 'configs/config.yaml'):
    """
    Main orchestration function for the L2 Candidate Ranking Engine.
    """
    # Load paths strictly from the config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    index_path = Path(config['paths']['index_path'])
    parquet_path = Path(config['paths']['parquet_path'])
    
    # Check for final count in config, default to 100 if not explicitly set yet
    final_count = config.get('retrieval', {}).get('final_count', 100)
    
    # --- 0. Pre-computation Failsafe ---
    if not index_path.exists() or not parquet_path.exists():
        logging.warning("Offline artifacts not found! Building them now from the raw --candidates file.")
        logging.warning("Note: This will take significant time on 100k rows.")
        try:
            artifact_dir = parquet_path.parent
            indexer = OfflineCandidateIndexer(
                input_path=candidates_path,
                output_dir=str(artifact_dir),
                model_name=config['model']['name'],
                parquet_filename=parquet_path.name,
                faiss_filename=index_path.name
            )
            indexer.run()
        except Exception as e:
            logging.error(f"Failed to build index: {e}")
            sys.exit(1)
    else:
        logging.info("Pre-computed fast artifacts found. Bypassing JSONL extraction.")
            
    logging.info("Starting Online L2 Ranking Engine...")
    
    # --- 1. Embed the Job Description ---
    jd_text = get_jd_text(config_path)
    jd_vector = generate_query_vector(jd_text, config_path)
    
    # --- 2. Semantic Retrieval (FAISS Search) ---
    top_candidates_df = search_candidates(jd_vector, config_path)
    
    # --- 3. Hard Filtering (Deterministic Drops) ---
    logging.info("Applying hard heuristic filters...")
    service_companies = ['tcs', 'wipro', 'infosys', 'cognizant']
    top_candidates_df = top_candidates_df[~top_candidates_df['current_company'].str.lower().isin(service_companies)]
    
    # --- 4. Behavioral Multipliers (L2 Ranking) ---
    logging.info("Applying behavioral math modifiers...")
    # (Assuming evaluation date is during the hackathon judging period)
    scored_df = apply_behavioral_math(top_candidates_df, evaluation_date_str='2026-06-25', config_path=config_path)
    
    # --- 5. Sort and Slice Top N ---
    logging.info(f"Sorting and selecting the Top {final_count} candidates...")
    final_candidates = scored_df.sort_values(by='final_score', ascending=False).head(final_count).copy()
    
    # --- 6. Generate Reasoning Strings ---
    logging.info("Generating dynamic reasoning...")
    final_candidates['reasoning'] = final_candidates.apply(generate_reasoning, axis=1)
    
    # --- 7. Format Output to strict Hackathon specs ---
    submission_df = format_submission_df(final_candidates)
    
    # --- 8. Save Output ---
    logging.info(f"Saving final ranked output to {output_path}...")
    submission_df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info("✅ Ranking Complete! Output is ready for submission.")

if __name__ == "__main__":
    # The required CLI Argument Parser for Stage 3 evaluation
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument('--candidates', required=True, help="Path to the raw candidates JSONL file")
    parser.add_argument('--out', required=True, help="Path to save the output CSV")
    
    args = parser.parse_args()
    
    # Execute the ranker using the command line arguments
    run_ranker(args.candidates, args.out)