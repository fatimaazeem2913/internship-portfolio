"""
tfidf_manual.py
---------------
Computes TF-IDF completely from scratch (pure Python) for 5 real sentences
from our Day 1 corpus, then verifies the result against sklearn's TfidfVectorizer.

IMPORTANT: sklearn's default TfidfVectorizer uses RAW term counts for TF
(not term count / document length, which is the "classic" textbook TF).
This script replicates sklearn's exact formula so the numbers match:

    tf(t, d)  = raw count of term t in document d
    idf(t)    = ln( (1 + n) / (1 + df(t)) ) + 1        [smooth_idf=True, sklearn default]
    tfidf(t,d)= tf(t,d) * idf(t)
    final vector for d = L2-normalized tfidf vector      [norm='l2', sklearn default]
"""

import re
import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# 5 real sentences pulled from the Day 1 scraped corpus (one from each domain, mixed)
SENTENCES = [
    "The global COVID pandemic caused severe social and economic disruption around the world",
    "Astronomers use powerful telescopes to search for exoplanets orbiting distant stars",
    "An exoplanet is a planet located outside our solar system orbiting another star",
    "Generators are iterators that do not store all values in memory",
    "The COVID pandemic led to travel restrictions and business closures worldwide",
]


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def compute_manual_tfidf(sentences):
    tokenized_docs = [tokenize(s) for s in sentences]
    n_docs = len(tokenized_docs)

    # Build vocabulary (sorted, matches sklearn's alphabetical default ordering)
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))

    # Document frequency: how many documents contain each term
    df = defaultdict(int)
    for doc in tokenized_docs:
        for word in set(doc):
            df[word] += 1

    # IDF using sklearn's smoothed formula
    idf = {word: math.log((1 + n_docs) / (1 + df[word])) + 1 for word in vocab}

    # Raw term frequency (count) per document
    tf_matrix = []
    for doc in tokenized_docs:
        counts = defaultdict(int)
        for word in doc:
            counts[word] += 1
        tf_matrix.append(counts)

    # TF-IDF = tf * idf, then L2 normalize each document vector
    tfidf_matrix = []
    for counts in tf_matrix:
        vec = [counts.get(word, 0) * idf[word] for word in vocab]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        tfidf_matrix.append(vec)

    return vocab, idf, tfidf_matrix


if __name__ == "__main__":
    vocab, idf, manual_matrix = compute_manual_tfidf(SENTENCES)

    print("=" * 90)
    print("MANUAL TF-IDF (from scratch)")
    print("=" * 90)
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Vocabulary: {vocab}\n")

    print("IDF scores (sample of 10 words):")
    for word in vocab[:10]:
        print(f"  {word:<15} idf = {idf[word]:.4f}")

    print("\nManual TF-IDF vectors (first 8 dims shown per sentence):")
    for i, vec in enumerate(manual_matrix):
        print(f"  Sentence {i+1}: {[round(v, 4) for v in vec[:8]]} ...")

    # --- Verify against sklearn ---
    sk_vectorizer = TfidfVectorizer(token_pattern=r"[a-z']+", lowercase=True)
    sk_matrix = sk_vectorizer.fit_transform(SENTENCES).toarray()
    sk_vocab = sk_vectorizer.get_feature_names_out()

    print("\n" + "=" * 90)
    print("SKLEARN TfidfVectorizer (verification)")
    print("=" * 90)
    print(f"\nsklearn vocabulary size: {len(sk_vocab)}")
    print(f"Vocabularies match: {list(sk_vocab) == vocab}")

    # Compare numerically
    import numpy as np
    manual_np = np.array(manual_matrix)
    diff = np.abs(manual_np - sk_matrix)
    max_diff = diff.max()

    print(f"\nMax absolute difference between manual and sklearn TF-IDF values: {max_diff:.10f}")
    print("MATCH CONFIRMED" if max_diff < 1e-9 else "MISMATCH — investigate")

    print("\nsklearn TF-IDF vectors (first 8 dims shown per sentence):")
    for i, vec in enumerate(sk_matrix):
        print(f"  Sentence {i+1}: {[round(v, 4) for v in vec[:8]]} ...")

    # Save results
    with open("outputs/tfidf_manual_vs_sklearn.txt", "w", encoding="utf-8") as f:
        f.write(f"Vocabulary ({len(vocab)} terms): {vocab}\n\n")
        f.write("IDF scores:\n")
        for word in vocab:
            f.write(f"  {word:<15} idf = {idf[word]:.4f}\n")
        f.write(f"\nMax absolute difference manual vs sklearn: {max_diff:.10f}\n")
        f.write("MATCH CONFIRMED\n" if max_diff < 1e-9 else "MISMATCH\n")
        f.write("\nFull manual TF-IDF matrix:\n")
        for i, vec in enumerate(manual_matrix):
            f.write(f"Sentence {i+1}: {[round(v, 4) for v in vec]}\n")

    print("\nSaved to outputs/tfidf_manual_vs_sklearn.txt")
