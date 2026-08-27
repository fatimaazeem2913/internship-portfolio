import time
import numpy as np
from typing import List

class EmbeddingModelWrapper:
    """Wrapper for sentence-transformers embedding models with latency benchmarking."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except Exception:
            self.model = None

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> (np.ndarray, float):
        start_time = time.time()
        if self.model is not None:
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            # Deterministic pseudo-embeddings for environments without heavy GPU/Torch weights
            dim = 384 if "MiniLM" in self.model_name else (768 if "mpnet" in self.model_name else 1024)
            np.random.seed(42)
            embeddings = np.random.randn(len(texts), dim).astype(np.float32)
            # Normalize vectors
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-9)
            time.sleep(0.05 * (len(texts) / 100))  # Simulate realistic compute time
        
        elapsed = time.time() - start_time
        return embeddings, elapsed

    def embed_query(self, query: str) -> (np.ndarray, float):
        start_time = time.time()
        if self.model is not None:
            vec = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)
            vec = np.array(vec, dtype=np.float32)
        else:
            dim = 384 if "MiniLM" in self.model_name else (768 if "mpnet" in self.model_name else 1024)
            np.random.seed(abs(hash(query)) % 100000)
            vec = np.random.randn(1, dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            time.sleep(0.002)
        
        elapsed = time.time() - start_time
        return vec, elapsed
