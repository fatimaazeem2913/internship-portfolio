"""
bow_extractor.py
----------------
A from-scratch Bag-of-Words (BoW) feature extractor.
No sklearn/CountVectorizer used — pure Python, so you can see exactly
what's happening under the hood.
"""

import re
from collections import defaultdict


def simple_sentence_split(text):
    """Very simple sentence splitter on . ! ?"""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def simple_word_tokenize(sentence):
    """Lowercase + extract word tokens only (no punctuation)."""
    return re.findall(r"[a-zA-Z']+", sentence.lower())


def build_vocabulary(sentences_tokens):
    """Builds a sorted vocabulary (unique words) across all sentences."""
    vocab = set()
    for tokens in sentences_tokens:
        vocab.update(tokens)
    return sorted(vocab)


def sentence_to_bow(tokens):
    """Turns a list of tokens into a {word: frequency} dictionary."""
    freq = defaultdict(int)
    for tok in tokens:
        freq[tok] += 1
    return dict(freq)


def build_bow_matrix(text):
    sentences = simple_sentence_split(text)
    sentences_tokens = [simple_word_tokenize(s) for s in sentences]
    vocab = build_vocabulary(sentences_tokens)

    bow_dicts = [sentence_to_bow(tokens) for tokens in sentences_tokens]

    # Also build a dense matrix form: rows = sentences, cols = vocab
    matrix = []
    for bow in bow_dicts:
        row = [bow.get(word, 0) for word in vocab]
        matrix.append(row)

    return sentences, vocab, bow_dicts, matrix


if __name__ == "__main__":
    with open("outputs/cleaned_corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Use just a slice for readability in the report (full corpus vocab would be huge)
    sample_text = (
        "Global markets surged as investors grew confident. "
        "Astronomers identified a rocky exoplanet in the habitable zone. "
        "Jordan fixed the tokenization bug but the lemmatizer still misbehaves. "
        "The lemmatizer bug confused irregular nouns."
    )

    sentences, vocab, bow_dicts, matrix = build_bow_matrix(sample_text)

    lines = ["=" * 90, "BAG-OF-WORDS (from scratch)", "=" * 90]
    lines.append(f"\nVocabulary ({len(vocab)} unique words):\n{vocab}\n")

    for i, (sent, bow) in enumerate(zip(sentences, bow_dicts)):
        lines.append(f"Sentence {i+1}: {sent}")
        lines.append(f"  BoW dict: {bow}\n")

    lines.append("Dense BoW matrix (rows = sentences, cols = vocab order above):")
    for i, row in enumerate(matrix):
        lines.append(f"  Sentence {i+1}: {row}")

    output = "\n".join(lines)
    print(output)

    with open("outputs/bow_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/bow_results.txt")
