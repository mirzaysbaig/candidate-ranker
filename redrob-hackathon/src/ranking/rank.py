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
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
    p_profile = Path(config['paths']['profile_index_path'])
    p_skills = Path(config['paths']['skills_index_path'])
    p_career = Path(config['paths']['career_index_path'])
    parquet_path = Path(config['paths']['parquet_path'])
    
    # Dynamic Stage Funnel Sizes
    stage2_count = config.get('retrieval', {}).get('stage2_hard_filter_count', 200)
    stage3_ce_count = config.get('retrieval', {}).get('stage3_ce_count', 100)
    stage4_final_count = config.get('retrieval', {}).get('stage4_final_count', 50)
    # --- 0. Pre-computation Failsafe ---
    if not p_profile.exists() or not p_skills.exists() or not p_career.exists() or not parquet_path.exists():
        logging.error("Offline Multi-FAISS artifacts not found! Run src.indexing.offline_indexer first.")
        sys.exit(1)
    else:
        logging.info("Pre-computed Multi-FAISS fast artifacts found. Bypassing JSONL extraction.")
            
    logging.info("Starting Online L2 Ranking Engine...")
    
    # --- 1. Embed the Job Description ---
    jd_text = get_jd_text(config_path)
    jd_vector = generate_query_vector(jd_text, config_path)
    
    # --- 2. Semantic Retrieval (FAISS Search) ---
    top_candidates_df = search_candidates(jd_text, jd_vector, config_path)
    
    # Save the intermediate FAISS ranking for debugging
    faiss_out_path = Path('data/output/faiss_ranking.csv')
    faiss_out_path.parent.mkdir(parents=True, exist_ok=True)
    top_candidates_df.to_csv(faiss_out_path, index=False)
    logging.info(f"Intermediate FAISS ranking saved to {faiss_out_path}")
    

    
    # --- 3. Dynamic Parser & Hard Filtering (Phase 2) ---
    logging.info("Parsing JD for dynamic constraints...")
    jd_rules = parse_jd_rules(jd_text)
    
    from src.models.scorer import apply_hard_filters
    logging.info("Applying deterministic hard filters (Phase 2)...")
    filtered_df = apply_hard_filters(top_candidates_df, jd_rules, config_path)
    
    # Calculate intermediate filtered score and export
    coherence_out_path = Path('data/output/coherence_output.csv')
    ce_out_path = Path('data/output/crossencoder_output.csv')
    filtered_df['filtered_score'] = (
        filtered_df['semantic_score'] * 
        filtered_df['coherence_mult'] * 
        filtered_df['experience_band_mult'] * 
        filtered_df['jd_hard_mult']
    )
    
    # Enforce Stage 2 Cut
    logging.info(f"Sorting and selecting the Top {stage2_count} candidates after Phase 2...")
    filtered_df_sorted = filtered_df.sort_values(by='filtered_score', ascending=False).head(stage2_count).copy()
    filtered_df_sorted.to_csv(coherence_out_path, index=False)
    logging.info(f"Data after Phase 2 (Hard Filters, Top {stage2_count}) saved to {coherence_out_path}")
    

    # --- 4. Cross-Encoder Deep Semantic Scoring (Stage 3) ---
    logging.info("Initializing Cross-Encoder Microservice...")
    from src.models.cross_encoder import CrossEncoderScorer
    ce_model_name = config.get('model', {}).get('cross_encoder_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    ce_scorer = CrossEncoderScorer(model_name=ce_model_name)
    
    # We need candidate texts. We extract the raw profiles from metadata
    logging.info("Preparing candidate texts for Cross-Encoder...")
    candidate_texts = []
    for _, row in filtered_df_sorted.iterrows():
        # Combine relevant context for the CE
        t = f"{row.get('all_job_titles', '')} {row.get('all_skills', '')}".strip()
        candidate_texts.append(t if t else "No profile")
        
    ce_scores = ce_scorer.score_candidates(jd_text, candidate_texts)
    
    # Inject the Cross-Encoder scores back into the dataframe
    filtered_df_sorted['ce_score'] = ce_scores
    
    logging.info(f"Sorting and selecting the Top {stage3_ce_count} candidates after Phase 3 (Cross-Encoder)...")
    ce_df_sorted = filtered_df_sorted.sort_values(by='ce_score', ascending=False).head(stage3_ce_count).copy()
    ce_df_sorted.to_csv(ce_out_path, index=False)
    logging.info(f"Data after Phase 3 (Cross-Encoder, Top {stage3_ce_count}) saved to {ce_out_path}")
    
    # --- 5. Behavioral Multipliers & Scorer (Stage 4) ---
    logging.info("Applying behavioral math modifiers (Phase 4)...")
    scored_df = apply_behavioral_math(
        ce_df_sorted, 
        jd_rules=jd_rules,
        evaluation_date_str='2026-06-25', 
        config_path=config_path
    )

    # --- 4.5. Normalize Scores (Strictly 0.5 to 1.0) ---
    logging.info("Normalizing final scores to a strict 0.5 - 1.0 scale...")
    min_score = scored_df['final_score'].min()
    max_score = scored_df['final_score'].max()
    if max_score > min_score:
        scored_df['final_score'] = 0.5 + 0.5 * (scored_df['final_score'] - min_score) / (max_score - min_score)
    else:
        scored_df['final_score'] = 1.0
    
    # --- 5. Sort and Slice Top N ---
    logging.info(f"Sorting and selecting the Top {stage4_final_count} candidates...")
    final_candidates = scored_df.sort_values(by='final_score', ascending=False).head(stage4_final_count).copy()
    
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