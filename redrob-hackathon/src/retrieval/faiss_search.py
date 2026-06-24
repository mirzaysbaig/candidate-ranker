import faiss
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Strictly loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def search_candidates(query_vector: np.ndarray, config_path: str = 'configs/config.yaml') -> pd.DataFrame:
    """
    Executes an ANN search against the FAISS database using paths and limits from config.yaml.
    """
    # STRICT READ from YAML
    config = load_config(config_path)
    index_path = config['paths']['index_path']
    parquet_path = config['paths']['parquet_path']
    top_k = config['retrieval']['top_k']
    
    if not Path(index_path).exists() or not Path(parquet_path).exists():
        logging.error("Offline artifacts missing! You must run src.indexing.offline_indexer first.")
        raise FileNotFoundError("FAISS index or Parquet metadata not found.")
        
    # 1. Load the FAISS Index
    logging.info(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))
    
    # 2. Execute the Vector Search
    logging.info(f"Scanning 100,000 vectors for the top {top_k} closest matches...")
    distances, indices = index.search(query_vector, k=top_k)
    
    matched_indices = indices[0]
    matched_scores = distances[0]
    
    # 3. Map back to Metadata
    logging.info(f"Loading candidate metadata from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Slice the massive 100k dataframe down to just the matches we found
    top_candidates = df.iloc[matched_indices].copy()
    
    # Inject the FAISS similarity score directly into the dataframe
    top_candidates['semantic_score'] = matched_scores
    
    return top_candidates