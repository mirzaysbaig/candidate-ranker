import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import yaml

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Strictly loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_jd_text(config_path: str = 'configs/config.yaml') -> str:
    """Reads the Job Description text using the path defined in config.yaml."""
    config = load_config(config_path)
    jd_path = config['paths']['jd_path']
    
    path = Path(jd_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        logging.warning(f"JD file not found at {jd_path}. Using a fallback AI Engineer intent.")
        return "Senior AI Engineer. Machine Learning Systems, Embeddings, Retrieval Augmented Generation (RAG), Ranking algorithms, LLMs, fine-tuning. Python, backend systems, product engineering."

def generate_query_vector(text: str, config_path: str = 'configs/config.yaml') -> np.ndarray:
    """
    Converts the text into a dense vector using the model defined in config.yaml.
    """
    config = load_config(config_path)
    model_name = config['model']['name']
    
    logging.info(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    logging.info("Generating dense vector for the Job Description...")
    vector = model.encode([text])
    vector_array = np.array(vector).astype('float32')
    
    # Normalize for Cosine Similarity
    faiss.normalize_L2(vector_array)
    
    return vector_array