"""
word2vec_analysis.py
----------------------
1. Tests word analogies (king - man + woman ~= queen style, adapted to our corpus vocabulary)
2. Runs nearest-neighbor queries
3. Re-measures the same 5 synonym pairs using Word2Vec and compares against TF-IDF scores
"""

from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load("outputs/word2vec_day2.model")
wv = model.wv


def safe_similarity(word_a, word_b):
    if word_a in wv and word_b in wv:
        return wv.similarity(word_a, word_b)
    return None


if __name__ == "__main__":
    lines = ["=" * 90, "WORD2VEC ANALYSIS: ANALOGIES, NEAREST NEIGHBORS, SYNONYM COMPARISON", "=" * 90]

    # --- 1. Analogies ---
    # NOTE: our corpus is only ~25k tokens (COVID + Exoplanet + StackOverflow),
    # so the classic king-man+woman=queen analogy won't work (those words barely
    # appear, if at all). We test analogies using words that ACTUALLY appear
    # frequently enough in our own corpus to have learned meaningful vectors.
    lines.append("\n--- 1. WORD ANALOGIES ---")
    lines.append("(Classic king-man+woman=queen doesn't apply -- our corpus is domain-specific")
    lines.append(" news/science/code text, not general fiction/dialogue with royalty words.")
    lines.append(" Testing analogies using vocabulary that actually appears in OUR corpus.)\n")

    analogy_tests = [
        (["planet", "star"], ["moon"]),          # planet:star :: moon:? (celestial relations)
        (["virus", "disease"], ["infection"]),    # virus:disease :: infection:?
        (["function", "return"], ["yield"]),      # code semantics
    ]

    for positive, negative in analogy_tests:
        available_pos = [w for w in positive if w in wv]
        available_neg = [w for w in negative if w in wv]
        lines.append(f"Analogy: {' + '.join(positive)} - {' - '.join(negative) if negative else '(none)'}")
        if len(available_pos) < len(positive) or len(available_neg) < len(negative):
            missing = set(positive + negative) - set(available_pos + available_neg)
            lines.append(f"  SKIPPED - word(s) not in vocabulary (too rare in our small corpus): {missing}")
            continue
        try:
            result = wv.most_similar(positive=available_pos, negative=available_neg, topn=3)
            lines.append(f"  Top 3 results: {result}")
        except Exception as e:
            lines.append(f"  Error: {e}")
        lines.append("")

    # --- 2. Nearest neighbor queries ---
    lines.append("\n--- 2. NEAREST NEIGHBOR QUERIES ---\n")
    query_words = ["pandemic", "planet", "function", "virus", "star", "code"]
    for word in query_words:
        if word in wv:
            neighbors = wv.most_similar(word, topn=5)
            lines.append(f"Nearest neighbors of '{word}':")
            for neighbor, score in neighbors:
                lines.append(f"    {neighbor:<20} similarity={score:.4f}")
        else:
            lines.append(f"'{word}' not in vocabulary (appeared fewer than min_count=2 times)")
        lines.append("")

    # --- 3. Synonym pair comparison: Word2Vec vs TF-IDF ---
    lines.append("\n--- 3. SYNONYM PAIR COMPARISON: WORD2VEC vs TF-IDF ---\n")
    lines.append("(Using single-word pairs so we can query Word2Vec's word-level vectors directly)\n")

    # Single-word synonym pairs, chosen to actually appear in our small corpus
    word_synonym_pairs = [
        ("virus", "pathogen"),
        ("disease", "illness"),
        ("planet", "world"),
        ("study", "research"),
        ("function", "method"),
    ]

    # These are the TF-IDF sentence-level results from Day 2 Task 3 (synonym_tfidf.py), for reference
    tfidf_reference_scores = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    lines.append(f"{'Pair':<30}{'Word2Vec similarity':<25}{'TF-IDF (sentence-level, Task 3)'}")
    lines.append("-" * 85)
    for (w1, w2), tfidf_score in zip(word_synonym_pairs, tfidf_reference_scores):
        sim = safe_similarity(w1, w2)
        sim_str = f"{sim:.4f}" if sim is not None else "N/A (word not in vocab)"
        lines.append(f"{w1 + ' / ' + w2:<30}{sim_str:<25}{tfidf_score:.4f}")

    lines.append("\nNote: Word2Vec similarity may still be low/moderate here because our training")
    lines.append("corpus is very small (~25k tokens) -- Word2Vec typically needs millions of tokens")
    lines.append("to learn rich semantic relationships. Word embeddings from models pretrained on")
    lines.append("billions of words (e.g., Google News 300d vectors) show MUCH stronger synonym")
    lines.append("similarity. This is an honest limitation of training on a small corpus, and is")
    lines.append("itself a useful lesson: embedding QUALITY scales with training DATA VOLUME.")

    output = "\n".join(lines)
    print(output)

    with open("outputs/word2vec_analysis_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/word2vec_analysis_results.txt")
