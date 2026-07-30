"""
word2vec_train.py
-------------------
Trains a Word2Vec skip-gram model (gensim) on our Day 1 scraped corpus.

CONCEPTUAL LINK TO LLM PRETRAINING:
Skip-gram's training objective is: given a CENTER word, predict the
CONTEXT words around it. This is a "predictive" task, not a counting
task — the model must adjust its internal weights to get better at
predicting, and in doing so, it learns dense vector representations
that capture meaning and usage patterns.

This is the direct conceptual ancestor of GPT-style pretraining, where
the objective is "predict the next token given everything before it."
Both are self-supervised: no manual labels needed, the text itself
provides the training signal. Word2Vec predicts context from a center
word (or center from context, in CBOW mode); GPT predicts the next
token from all previous tokens. Different direction, same core idea:
learn meaning by learning to predict.
"""

import re
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec


def load_and_tokenize_corpus(path="data/input_corpus.txt"):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    for marker in ["### DOMAIN: NEWS ###", "### DOMAIN: SCIENCE ###", "### DOMAIN: DIALOGUE ###"]:
        raw = raw.replace(marker, "")

    sentences = sent_tokenize(raw)
    tokenized_sentences = []
    for sent in sentences:
        tokens = re.findall(r"[a-z']+", sent.lower())
        if len(tokens) >= 3:  # skip degenerate tiny sentences
            tokenized_sentences.append(tokens)
    return tokenized_sentences


if __name__ == "__main__":
    tokenized_sentences = load_and_tokenize_corpus()
    print(f"Training on {len(tokenized_sentences)} tokenized sentences.")
    total_tokens = sum(len(s) for s in tokenized_sentences)
    print(f"Total tokens: {total_tokens}")

    # sg=1 -> skip-gram (predict context from center word)
    # sg=0 would be CBOW (predict center from context) -- the reverse direction
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=100,      # dimensionality of each word's embedding vector
        window=5,              # how many words on each side count as "context"
        min_count=2,            # ignore words appearing fewer than 2 times
        sg=1,                    # 1 = skip-gram, 0 = CBOW
        epochs=50,                # multiple passes over the data since our corpus is small
        seed=42,
        workers=1,
    )

    model.save("outputs/word2vec_day2.model")

    vocab_size = len(model.wv.key_to_index)
    print(f"\nVocabulary size learned: {vocab_size} unique words")
    print(f"Embedding dimensionality: {model.wv.vector_size}")

    # Show a sample embedding vector (first 10 dims) for a common word
    sample_word = "pandemic" if "pandemic" in model.wv else list(model.wv.key_to_index.keys())[0]
    print(f"\nSample embedding vector for '{sample_word}' (first 10 of {model.wv.vector_size} dims):")
    print(model.wv[sample_word][:10])

    with open("outputs/word2vec_training_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Sentences trained on: {len(tokenized_sentences)}\n")
        f.write(f"Total tokens: {total_tokens}\n")
        f.write(f"Vocabulary size learned: {vocab_size}\n")
        f.write(f"Embedding dimensionality: {model.wv.vector_size}\n")
        f.write(f"Training mode: Skip-gram (sg=1)\n")
        f.write(f"Window size: 5, Epochs: 50, Min count: 2\n")

    print("\nModel saved to outputs/word2vec_day2.model")
    print("Summary saved to outputs/word2vec_training_summary.txt")
