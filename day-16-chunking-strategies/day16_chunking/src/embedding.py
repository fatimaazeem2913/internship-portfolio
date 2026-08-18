"""
embedding.py
------------
Provides the embedding function semantic chunking needs, following the
exact real-model-plus-honest-fallback pattern established in Day 15:
sentence-transformers as the real, primary path; a network-free TF-IDF
fallback (scikit-learn) for this sandbox, always clearly logged as such,
never silently substituted without saying so.
"""

from __future__ import annotations

_model = None
_backend = None


def _load():
    global _model, _backend
    if _model is not None or _backend is not None:
        return

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _backend = "sentence-transformers"
        print("[embedding] Loaded real model: all-MiniLM-L6-v2")
    except Exception as exc:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print(
            f"[embedding] WARNING: could not load all-MiniLM-L6-v2 "
            f"({type(exc).__name__}: huggingface.co unreachable in this sandbox). "
            f"Falling back to local TF-IDF embedder for this run (see Day 15's "
            f"embedding.py for the original documented instance of this issue)."
        )
        _model = TfidfVectorizer()
        _backend = "tfidf-fallback"


def get_backend() -> str:
    _load()
    return _backend


def embed_texts(texts: list[str]) -> list[list[float]]:
    import numpy as np

    _load()
    if _backend == "sentence-transformers":
        return _model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    # TF-IDF fallback: fit fresh each call for simplicity here, since
    # semantic chunking always embeds one document's own sentences together
    # in a single call, unlike Day 15's persistent query/corpus split.
    matrix = _model.fit_transform(texts).toarray()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).tolist()
