"""
ngram_word_vectors.py
------------------------
Builds a simple DISTRIBUTIONAL word vector from the bigram model: each word's
"vector" is its probability distribution over what words follow it,
P(next_word | this_word), taken across the WHOLE corpus. This is a classic
count-based distributional representation (the same core idea behind old-school
distributional semantics, before Word2Vec) -- and, like TF-IDF and Word2Vec,
it is STATIC: one fixed vector per word, aggregated across every occurrence
in the corpus, regardless of which specific sentence you're asking about.

Used here to add an "n-gram" column to the Day 3 comparison table.
"""

import re
import math
from collections import defaultdict, Counter
from nltk.tokenize import sent_tokenize


def load_corpus_tokens(path="data/input_corpus.txt"):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    for marker in ["### DOMAIN: NEWS ###", "### DOMAIN: SCIENCE ###", "### DOMAIN: DIALOGUE ###"]:
        raw = raw.replace(marker, "")
    sentences = sent_tokenize(raw)
    all_tokens = []
    for sent in sentences:
        tokens = re.findall(r"[a-z']+", sent.lower())
        if len(tokens) >= 2:
            all_tokens.append(tokens)
    return all_tokens


def build_bigram_distributions(sentences):
    """For every word, build a Counter of what words follow it anywhere in the corpus."""
    follow_counts = defaultdict(Counter)
    vocab = set()
    for tokens in sentences:
        vocab.update(tokens)
        for i in range(len(tokens) - 1):
            follow_counts[tokens[i]][tokens[i + 1]] += 1
    return follow_counts, sorted(vocab)


def word_vector(word, follow_counts, vocab):
    """Dense vector: P(next_word | word) across the full vocabulary."""
    counts = follow_counts.get(word, Counter())
    total = sum(counts.values())
    if total == 0:
        return [0.0] * len(vocab)
    return [counts.get(v, 0) / total for v in vocab]


def cosine_sim(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


if __name__ == "__main__":
    sentences = load_corpus_tokens()
    follow_counts, vocab = build_bigram_distributions(sentences)
    print(f"Built bigram distributions over {len(vocab)} vocabulary words.\n")

    # Same synonym pairs tested with TF-IDF and Word2Vec in Day 2
    synonym_pairs = [
        ("virus", "pathogen"),
        ("disease", "illness"),
        ("planet", "world"),
        ("study", "research"),
        ("function", "method"),
    ]

    lines = ["=" * 90, "N-GRAM DISTRIBUTIONAL WORD VECTORS: synonym similarity", "=" * 90]
    lines.append("\n(Vector for each word = its P(next_word | word) distribution, aggregated")
    lines.append(" across the WHOLE corpus -- a static, count-based representation.)\n")

    results = {}
    for w1, w2 in synonym_pairs:
        if w1 in vocab and w2 in vocab:
            v1 = word_vector(w1, follow_counts, vocab)
            v2 = word_vector(w2, follow_counts, vocab)
            sim = cosine_sim(v1, v2)
            results[(w1, w2)] = sim
            lines.append(f"{w1} / {w2}: cosine similarity = {sim:.4f}")
        else:
            missing = [w for w in (w1, w2) if w not in vocab]
            results[(w1, w2)] = None
            lines.append(f"{w1} / {w2}: N/A (word(s) not in vocabulary: {missing})")

    # Polysemy test: "light" compared with itself -- since this representation is
    # aggregated across the WHOLE corpus (not per-sentence), it is, by construction,
    # incapable of distinguishing the two senses -- exactly like Word2Vec.
    lines.append("\n--- Polysemy check: 'light' (n-gram distributional vector) ---\n")
    if "light" in vocab:
        v_light = word_vector("light", follow_counts, vocab)
        self_sim = cosine_sim(v_light, v_light)
        lines.append(f"'light' vector compared with itself: cosine similarity = {self_sim:.6f}")
        lines.append("(This will always be 1.0 -- there is only ONE 'light' vector in this")
        lines.append(" representation, aggregated across every sentence it appeared in. Just")
        lines.append(" like Word2Vec, the n-gram distributional vector cannot distinguish the")
        lines.append(" idiomatic sense of 'light' from the literal sense -- it has no mechanism")
        lines.append(" to look at a SPECIFIC sentence's context at lookup time.)")
    else:
        lines.append("'light' not found in vocabulary.")

    output = "\n".join(lines)
    print(output)

    with open("outputs/ngram_word_vector_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/ngram_word_vector_results.txt")
