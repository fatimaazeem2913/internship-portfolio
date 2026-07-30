"""
tfidf_retrieval.py
-------------------
Builds a TF-IDF cosine-similarity retrieval system: given a query, rank
all corpus sentences by how relevant they are to the query.
"""

import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_corpus_sentences(path="data/input_corpus.txt"):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    for marker in ["### DOMAIN: NEWS ###", "### DOMAIN: SCIENCE ###", "### DOMAIN: DIALOGUE ###"]:
        raw = raw.replace(marker, "")
    sentences = sent_tokenize(raw)
    # Keep reasonably sized, clean-ish sentences
    sentences = [s.strip() for s in sentences if 30 < len(s.strip()) < 300]
    return sentences


def build_retrieval_system(sentences):
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    doc_vectors = vectorizer.fit_transform(sentences)
    return vectorizer, doc_vectors


def search(query, vectorizer, doc_vectors, sentences, top_k=5):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, doc_vectors).flatten()
    ranked_idx = scores.argsort()[::-1][:top_k]
    return [(sentences[i], scores[i]) for i in ranked_idx]


if __name__ == "__main__":
    sentences = load_corpus_sentences()
    print(f"Loaded {len(sentences)} sentences into the retrieval corpus.\n")

    vectorizer, doc_vectors = build_retrieval_system(sentences)

    queries = [
        "vaccines and disease prevention",
        "planets orbiting stars in space",
        "python generator functions and memory",
    ]

    lines = ["=" * 90, "TF-IDF COSINE SIMILARITY RETRIEVAL SYSTEM", "=" * 90]

    for query in queries:
        results = search(query, vectorizer, doc_vectors, sentences, top_k=3)
        lines.append(f"\nQUERY: \"{query}\"")
        for rank, (sent, score) in enumerate(results, 1):
            lines.append(f"  [{rank}] (score={score:.4f}) {sent[:150]}")

    output = "\n".join(lines)
    print(output)

    with open("outputs/tfidf_retrieval_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/tfidf_retrieval_results.txt")
