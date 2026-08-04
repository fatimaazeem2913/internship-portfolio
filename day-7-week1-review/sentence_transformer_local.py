"""
sentence_transformer_local.py
--------------------------------
REFERENCE IMPLEMENTATION: the real all-MiniLM-L6-v2 embedding retrieval,
exactly as specified in the task. Run this LOCALLY with internet access --
this sandboxed verification environment cannot download models from
Hugging Face (confirmed directly, same restriction as Days 3 and 5).

This produces the SAME retrieval methodology as retrieval_comparison.py's
Method 3 (embed query + chunks, rank by cosine similarity) -- only the
embedding model differs: a real 384-dimensional sentence-transformer model
instead of the averaged-Word2Vec substitute used for in-sandbox verification.

SETUP (run on your own machine):
    pip install sentence-transformers
    python3 sentence_transformer_local.py
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

QUERY = "Who is the president of Pakistan?"


def load_corpus(path="data/corpus.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def embedding_retrieval_real(query, corpus, model):
    """
    Encodes the query and every chunk's RAW text (sentence-transformers
    models expect natural, un-stemmed text -- unlike BoW/TF-IDF, they do NOT
    benefit from stopword removal/lemmatization, since they were trained on
    natural sentences and their contextual understanding already handles
    function words appropriately).
    """
    texts = [c["raw_text"] for c in corpus]
    doc_embeddings = model.encode(texts)
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings).flatten()
    best_idx = int(np.argmax(scores))
    return corpus[best_idx], float(scores[best_idx]), scores


if __name__ == "__main__":
    print("Loading all-MiniLM-L6-v2 (downloads ~80MB on first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    corpus = load_corpus()
    print(f"Query: \"{QUERY}\"")
    print(f"Corpus size: {len(corpus)} chunks\n")

    chunk, score, all_scores = embedding_retrieval_real(QUERY, corpus, model)

    print(f"Top-1 chunk_id: {chunk['chunk_id']} (source: {chunk['source']})")
    print(f"Cosine similarity score: {score:.4f}")
    print(f"Text: {chunk['raw_text'][:200]}...")

    print("\nAll scores, ranked:")
    ranked = sorted(zip(corpus, all_scores), key=lambda x: -x[1])
    for c, s in ranked[:5]:
        print(f"  chunk_id={c['chunk_id']:<3} source={c['source']:<18} score={s:.4f}")
