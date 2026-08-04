"""
shared_utils.py
------------------
REFACTORING DELIVERABLE for Day 7: consolidates functions that were
duplicated with only minor variations across Days 1-6's individual project
folders into a single, well-documented, tested module.

Functions consolidated here, and where they previously lived in duplicate
or near-duplicate form:
    - tokenize()              : Day 1 (clean_text.py), Day 2 (multiple files),
                                 Day 3 (ngram_model.py), Day 5, Day 7
    - clean_text()             : Day 1 (clean_text.py), Day 7 (extract_chunk_clean.py)
    - cosine_similarity_manual(): Day 2 (synonym_tfidf.py), Day 3
                                 (ngram_word_vectors.py) -- both had their
                                 own separate, near-identical implementations
    - softmax()                : Day 4 (scaled_dot_product_attention.py)
    - load_json() / save_json(): repeated boilerplate across Days 2,3,5,6,7

REFACTORING PRINCIPLE APPLIED: each function here has ONE canonical
implementation, a docstring specifying its exact contract (inputs, outputs,
edge cases), and is meant to be imported rather than re-implemented by any
script that needs it going forward. Naming follows one consistent
convention: snake_case functions, verb-first names (tokenize, clean_text,
not text_cleaner or TextCleaner).
"""

import re
import json
import math
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))


def tokenize(text):
    """
    Tokenize text into lowercase alphabetic word tokens.

    Args:
        text (str): raw input text.

    Returns:
        list[str]: lowercase tokens, apostrophes preserved (e.g. "don't"),
        all other punctuation and digits stripped.

    Used by: Day 1 cleaning pipeline, Day 2 TF-IDF/Word2Vec preprocessing,
    Day 3 n-gram models, Day 5 next-token demos, Day 7 retrieval.
    """
    return re.findall(r"[a-z']+", text.lower())


def clean_text(text, remove_stopwords=True, lemmatize=True):
    """
    Full cleaning pipeline: lowercase, strip non-alphabetic characters,
    tokenize, optionally remove stopwords, optionally lemmatize.

    Args:
        text (str): raw input text.
        remove_stopwords (bool): if True, remove standard English stopwords.
        lemmatize (bool): if True, lemmatize each remaining token.

    Returns:
        str: cleaned text, tokens space-joined.

    Used by: Day 1 (clean_text.py -- originally implemented inline),
    Day 7 (extract_chunk_clean.py -- can now import this instead of
    re-implementing it).
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = re.findall(r"[a-z]+", text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    tokens = [t for t in tokens if len(t) > 1]

    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def cosine_similarity_manual(vec1, vec2):
    """
    Compute cosine similarity between two equal-length numeric vectors,
    from scratch (no sklearn/numpy dependency) -- verified in Day 2/3 to
    match sklearn's implementation to within floating-point precision.

    Args:
        vec1, vec2: iterables of numbers, equal length.

    Returns:
        float: cosine similarity, typically in [0, 1] for non-negative
        vectors like TF-IDF or count vectors. Returns 0.0 if either vector
        has zero magnitude, to avoid division by zero.

    Used by: Day 2 (synonym_tfidf.py), Day 3 (ngram_word_vectors.py) --
    both previously had their own separate, near-identical implementations.
    """
    vec1, vec2 = list(vec1), list(vec2)
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def softmax(scores):
    """
    Numerically stable softmax over a list of raw scores.

    Args:
        scores (list[float]): raw scores (logits).

    Returns:
        list[float]: probabilities summing to 1.0.

    Subtracts the max score before exponentiating to prevent overflow
    (exp(1000) overflows; exp(1000-1000)=exp(0)=1 does not) -- softmax is
    invariant to constant shifts, so this is mathematically identical to
    the naive version while being numerically safe.

    Used by: Day 4 (scaled_dot_product_attention.py) uses a NumPy 2D
    version for matrices; this scalar-list version is for simpler
    single-distribution use (Day 5 n-gram distributions, Day 6 sampling).
    """
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def load_json(path):
    """
    Load and parse a JSON file.

    Args:
        path (str): file path.

    Returns:
        dict or list: parsed JSON content.

    Used by: every day from Day 2 onward -- previously each script
    repeated the same open/json.load boilerplate independently.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path, indent=2):
    """
    Serialize data to a JSON file.

    Args:
        data (dict or list): data to serialize.
        path (str): output file path.
        indent (int): pretty-print indentation level.

    Used by: every day from Day 2 onward -- previously each script
    repeated the same open/json.dump boilerplate independently.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def word_frequency(tokens):
    """
    Build a word-frequency Counter from a list of tokens.

    Args:
        tokens (list[str]): tokenized words.

    Returns:
        collections.Counter: {word: count} mapping.

    Used by: Day 1 (bow_extractor.py), Day 3 (ngram_model.py context
    counting), Day 7 -- previously each re-implemented this via manual
    dict-building loops instead of using Counter directly.
    """
    return Counter(tokens)


if __name__ == "__main__":
    print("Running shared_utils.py self-tests...")

    assert tokenize("Hello, World! Don't panic.") == ["hello", "world", "don't", "panic"]
    print("  tokenize() OK")

    cleaned = clean_text("The Quick Brown Foxes are Running!")
    print(f"  clean_text() OK -> '{cleaned}'")
    assert "the" not in cleaned.split()

    sim = cosine_similarity_manual([1, 0, 1], [1, 0, 1])
    assert abs(sim - 1.0) < 1e-9
    sim_orth = cosine_similarity_manual([1, 0], [0, 1])
    assert abs(sim_orth - 0.0) < 1e-9
    print("  cosine_similarity_manual() OK")

    probs = softmax([1.0, 2.0, 3.0])
    assert abs(sum(probs) - 1.0) < 1e-9
    print("  softmax() OK")

    freq = word_frequency(["a", "b", "a", "c", "a"])
    assert freq["a"] == 3
    print("  word_frequency() OK")

    print("\nAll self-tests passed.")
