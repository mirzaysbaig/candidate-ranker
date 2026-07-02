import faiss
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
import pickle

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Strictly loads the YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_all_scores(index, query_vector: np.ndarray) -> np.ndarray:
    """Queries FAISS for all candidates and returns an array of scores ordered by candidate index."""
    ntotal = index.ntotal
    distances, indices = index.search(query_vector, k=ntotal)
    
    # distances[0] and indices[0] are sorted by score. 
    # We want to re-order them so score_array[i] corresponds to candidate index i.
    score_array = np.zeros(ntotal, dtype=np.float32)
    score_array[indices[0]] = distances[0]
    return score_array

def search_candidates(jd_text: str, query_vector: np.ndarray, config_path: str = 'configs/config.yaml') -> pd.DataFrame:
    """
    Executes an ANN search against the THREE FAISS databases using paths and limits from config.yaml.
    Applies On-The-Fly (OTF) weighting to compute the base semantic score.
    """
    # STRICT READ from YAML
    config = load_config(config_path)
    p_profile = config['paths']['profile_index_path']
    p_skills = config['paths']['skills_index_path']
    p_career = config['paths']['career_index_path']
    parquet_path = config['paths']['parquet_path']
    p_bm25 = config['paths'].get('bm25_index_path', 'data/processed/bm25_index.pkl')
    
    stage1_count = config.get('retrieval', {}).get('stage1_l1_count', 400)
    weights = config['retrieval']['multi_vector_weights']
    
    w_profile = weights.get('profile', 0.20)
    w_skills = weights.get('skills', 0.30)
    w_career = weights.get('career', 0.50)
    
    if not Path(p_profile).exists() or not Path(parquet_path).exists():
        logging.error("Offline artifacts missing! You must run src.indexing.offline_indexer first.")
        raise FileNotFoundError("FAISS indices or Parquet metadata not found.")
        
    # 1. Load the FAISS Indices
    logging.info(f"Loading Profile FAISS index...")
    idx_profile = faiss.read_index(str(p_profile))
    
    logging.info(f"Loading Skills FAISS index...")
    idx_skills = faiss.read_index(str(p_skills))
    
    logging.info(f"Loading Career FAISS index...")
    idx_career = faiss.read_index(str(p_career))    
    logging.info(f"Loading BM25 Sparse index...")
    with open(p_bm25, 'rb') as f:
        bm25_index = pickle.load(f)
    
    # 2. Execute Vector Search on all candidates
    logging.info(f"Scoring all vectors across 3 indices for perfect OTF weighting...")
    scores_profile = get_all_scores(idx_profile, query_vector)
    scores_skills = get_all_scores(idx_skills, query_vector)
    scores_career = get_all_scores(idx_career, query_vector)    
    # 3. Execute Sparse Search (BM25)
    jd_tokens = jd_text.lower().split()
    scores_bm25 = bm25_index.get_scores(jd_tokens)
    
    # Normalize BM25 scores (0.0 to 1.0) to mix with FAISS
    if scores_bm25.max() > 0:
        scores_bm25 = scores_bm25 / scores_bm25.max()
    
    # 3. Apply On-The-Fly Dense Weighting
    w_bm25 = weights.get('bm25', 0.10)
    base_semantic_scores = (
        (scores_bm25 * w_bm25) +
        (scores_profile * w_profile) + 
        (scores_skills * w_skills) + 
        (scores_career * w_career)
    )
    # 4. Semantic Delta (Keyword Stuffer) Penalty
    # This must happen HERE across the entire 100k array so stuffers don't pollute the Top K.
    sc_w = config.get('scoring_weights', {})
    high_skill = sc_w.get('coherence_high_skill_threshold', 0.65)
    delta_thresh = sc_w.get('coherence_delta_threshold', 0.30)
    partial_thresh = sc_w.get('coherence_partial_threshold', 0.20)
    
    mult_penalty = sc_w.get('coherence_penalty_multiplier', 0.10)
    mult_partial = sc_w.get('coherence_partial_multiplier', 0.60)
    mult_neutral = sc_w.get('coherence_neutral_multiplier', 1.00)
    
    delta = scores_skills - scores_career
    
    # Initialize multipliers array with neutral
    coherence_mults = np.full_like(base_semantic_scores, mult_neutral)
    
    heavy_stuffer_mask = (scores_skills > high_skill) & (delta > delta_thresh)
    partial_stuffer_mask = (scores_skills > high_skill) & (delta > partial_thresh) & ~heavy_stuffer_mask
    
    coherence_mults[heavy_stuffer_mask] = mult_penalty
    coherence_mults[partial_stuffer_mask] = mult_partial
    
    # Final adjusted score for ranking
    adjusted_semantic_scores = base_semantic_scores * coherence_mults
    
    # 5. Find the Top K indices based on the ADJUSTED combined score
    # argsort sorts ascending, so we take the last top_k and reverse
    top_indices = np.argsort(adjusted_semantic_scores)[-stage1_count:][::-1]
    
    # 6. Map back to Metadata
    logging.info(f"Loading candidate metadata from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Slice the massive 100k dataframe down to just the matches we found
    top_candidates = df.iloc[top_indices].copy()
    
    # Inject the individual and combined FAISS similarity scores directly into the dataframe
    top_candidates['sim_bm25'] = scores_bm25[top_indices]
    top_candidates['sim_profile'] = scores_profile[top_indices]
    top_candidates['sim_skills'] = scores_skills[top_indices]
    top_candidates['sim_career'] = scores_career[top_indices]
    top_candidates['semantic_score'] = base_semantic_scores[top_indices]
    top_candidates['coherence_mult'] = coherence_mults[top_indices]
    
    return top_candidates