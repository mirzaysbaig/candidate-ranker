import numpy as np
import logging
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CrossEncoderScorer:
    """
    Stage 3: Enterprise Semantic Reranker.
    Takes a narrowed list of candidates and pairs them with the JD for deep contextual scoring.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        logging.info(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        
    def score_candidates(self, jd_text: str, candidate_texts: list[str]) -> np.ndarray:
        """
        Calculates the deep semantic fit between the Job Description and the candidates.
        """
        logging.info(f"Passing {len(candidate_texts)} candidates to the Cross-Encoder...")
        
        # The Cross-Encoder requires data in the format: [(Query, Document), (Query, Document), ...]
        sentence_pairs = [[jd_text, text] for text in candidate_texts]
        
        # This is the heavy computation block
        raw_scores = self.model.predict(sentence_pairs, show_progress_bar=True)
        
        # Cross encoders often return unbounded "logits" (e.g. -4.5 to +8.2). 
        # We apply a sigmoid math function to crush them into a strict 0.0 -> 1.0 percentage scale.
        normalized_scores = 1 / (1 + np.exp(-raw_scores))
        
        return normalized_scores
