import argparse
import logging
import sys
from pathlib import Path
import yaml

# Import our custom enterprise microservices
from src.embeddings.embedding_service import get_jd_text, generate_query_vector
from src.utils.jd_parser import parse_jd_rules
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
                faiss_filename=index_path.name,
                indexer_config=config.get('indexer', {}),
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
    
    # --- 3. Dynamic Parser & Hard Filtering (Deterministic Drops) ---
    logging.info("Parsing JD for dynamic constraints...")
    jd_rules = parse_jd_rules(jd_text, config.get('jd_parsing', {}))
    
    # --- 4. Behavioral Multipliers & Scorer (L2 Ranking) ---
    logging.info("Applying behavioral math modifiers and hard filters...")
    # (Assuming evaluation date is during the hackathon judging period)
    scored_df = apply_behavioral_math(
        top_candidates_df, 
        jd_rules=jd_rules,
        evaluation_date_str='2026-06-25', 
        config_path=config_path
    )
    
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
    # Command line arguments are optional. If missing, we pull from config.yaml
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument('--candidates', required=False, help="Path to the raw candidates JSONL file")
    parser.add_argument('--out', required=False, help="Path to save the output CSV")
    parser.add_argument('--config', default='configs/config.yaml', help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config to get default paths if arguments aren't provided
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    # Use CLI args if provided, otherwise fallback to config.yaml
    candidates_path = args.candidates if args.candidates else config['paths']['raw_candidates']
    output_path = args.out if args.out else config['paths']['output_csv']
    
    # Execute the ranker
    run_ranker(candidates_path, output_path, args.config)