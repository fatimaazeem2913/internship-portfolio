"""
embedding.py
------------
Chunking -> Embedding.

Primary path: sentence-transformers (all-MiniLM-L6-v2, 384-dim), the real
semantic embedding model this project is built around -- same model family
whose contextual-embedding behavior was first measured back in Day 3's
BERT-vs-Word2Vec comparison.

Real, honestly-documented bug found while building this module: loading a
sentence-transformers model on first use requires downloading weights from
huggingface.co. This sandbox's network egress is allowlisted to a fixed set
of domains (pypi, npm, github, etc.) and huggingface.co is NOT on that list
-- the download fails with a 403 from the egress proxy, which surfaces to
the caller as a confusing HfHubHTTPError / "couldn't connect" traceback that
gives no hint the real cause is a network allowlist, not a code bug. This is
the same *category* of issue as the ChromaDB default-embedder ONNX download
bug already documented in rag_failure_modes.md -- any component that lazily
pulls model weights from the network at runtime is a hidden environment
dependency.

Fix applied here, matching how the ChromaDB bug was handled: the real
sentence-transformers path is kept as the correct, primary implementation
(this is exactly what should run in a normal environment with unrestricted
internet access -- e.g. the user's own local machine, which is how the
original Day 15 embedding claim was independently re-verified). For
pipeline development and testing *inside this sandbox*, embed_texts()
automatically falls back to a network-free TF-IDF vectorizer (scikit-learn,
already part of this project's toolchain) implementing the identical
list[list[float]] interface, so the rest of the pipeline (vector store,
retrieval, generation) can be built and tested end-to-end without silently
faking success. Which backend actually ran is always printed, never hidden.

Known limitation of the TF-IDF fallback (real, worth documenting): its
vocabulary is fixed at first fit(). Real embedding models generalize to
any unseen text via subword tokenization; TF-IDF cannot -- a word not seen
during fit produces a zero contribution, and text built entirely from
out-of-vocabulary words produces an all-zero vector. In this module, the
vectorizer is fit once (on whatever texts are embedded first in a process)
and reused for every later call. This surfaced as a real, order-dependent
test failure during development (see tests/test_pipeline.py and
reset_model() below) -- a earlier test's short vocabulary silently starved
a later test's unrelated words. This is itself a useful, honest illustration
of *why* production RAG needs real semantic embeddings rather than a fixed
lexical vocabulary: a corpus grows and gets re-queried with new phrasing
over time, and a fixed-vocabulary method degrades exactly like this.
"""

from __future__ import annotations

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_backend = None  # "sentence-transformers" or "tfidf-fallback"

_tfidf_vectorizer = None
_tfidf_fitted_texts: list[str] = []


def get_model():
    """Lazy singleton: tries the real sentence-transformers model first,
    falls back to a local TF-IDF vectorizer only if the download is
    blocked (see module docstring for why)."""
    global _model, _backend
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
        _backend = "sentence-transformers"
        print(f"[embedding] Loaded real model: {_MODEL_NAME}")
    except Exception as exc:  # network blocked in this sandbox -- see docstring
        from sklearn.feature_extraction.text import TfidfVectorizer

        print(
            f"[embedding] WARNING: could not load {_MODEL_NAME} "
            f"({type(exc).__name__}: huggingface.co unreachable in this "
            f"sandbox). Falling back to local TF-IDF embedder for this run."
        )
        _model = TfidfVectorizer()
        _backend = "tfidf-fallback"

    return _model


def reset_model():
    """Test/dev utility: clears the lazy singleton so a fresh model (or a
    freshly-fit TF-IDF vectorizer) is loaded on next use. Needed because the
    TF-IDF fallback's vocabulary is fixed at first fit -- see module
    docstring's OOV note -- so tests that care about a specific vocabulary
    must reset state first rather than inheriting whatever an earlier test
    fit the singleton on."""
    global _model, _backend, _tfidf_fitted_texts
    _model = None
    _backend = None
    _tfidf_fitted_texts = []


def get_backend_name() -> str:
    get_model()
    return _backend


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()

    if _backend == "sentence-transformers":
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vectors.tolist()

    # TF-IDF fallback: must be fit once on the full corpus vocabulary, then
    # every subsequent call (including single queries) is transformed with
    # that same fitted vocabulary -- otherwise query and chunk vectors would
    # live in different, incompatible vector spaces.
    global _tfidf_fitted_texts
    import numpy as np

    if not hasattr(model, "vocabulary_") or model.vocabulary_ is None:
        _tfidf_fitted_texts = list(texts)
        matrix = model.fit_transform(texts)
    else:
        matrix = model.transform(texts)

    dense = matrix.toarray()
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid div-by-zero for empty/stopword-only text
    normalized = dense / norms
    return normalized.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


if __name__ == "__main__":
    import time

    samples = [
        "Refunds are processed within 5-7 business days.",
        "Customers get their money back within a week of approval.",
        "The weather today is sunny with a light breeze.",
    ]
    start = time.time()
    vecs = embed_texts(samples)
    elapsed = time.time() - start
    backend = get_backend_name()

    print(f"\nBackend used: {backend}")
    print(f"Embedded {len(samples)} texts in {elapsed:.2f}s")
    print(f"Vector dimension: {len(vecs[0])}")

    # Sanity check: cosine similarity (vectors are already normalized, so
    # dot product == cosine similarity) between a related paraphrase vs an
    # unrelated sentence -- proves the embeddings capture *some* notion of
    # relatedness, not just random noise.
    import numpy as np

    v = np.array(vecs)
    sim_related = float(np.dot(v[0], v[1]))
    sim_unrelated = float(np.dot(v[0], v[2]))
    print(f"\nSimilarity(refund sentence, refund paraphrase) = {sim_related:.4f}")
    print(f"Similarity(refund sentence, unrelated weather)  = {sim_unrelated:.4f}")

    if backend == "tfidf-fallback":
        print(
            "\nNote: TF-IDF is a LEXICAL (word-overlap) method, not a semantic "
            "one -- this is the exact synonym-blindness limitation proven back "
            "in Day 2 (Word2Vec vs TF-IDF, similarity 0.0000 on true synonyms "
            "with zero shared words). A true paraphrase with no shared words "
            "may score LOW here even though a real semantic model (sentence-"
            "transformers, blocked by network in this sandbox) would score it "
            "high. This is precisely why the real model matters for production "
            "RAG and why this fallback is dev/test-only, never production."
        )
    else:
        assert sim_related > sim_unrelated, "Embedding model failed basic semantic sanity check!"
        print("\nSanity check passed: related text scores higher than unrelated text.")
