from sentence_transformers import SentenceTransformer
import numpy as np


class Vectorizer:
    """
    Creates embeddings for resume and job descriptions.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        """
        return self.model.encode(text)

    def embed_texts(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """
        return self.model.encode(texts, show_progress_bar=True)
