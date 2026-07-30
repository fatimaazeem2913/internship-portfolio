"""
ngram_model.py
---------------
Implements bigram and trigram language models with Laplace (add-1) smoothing,
trained on the real Day 1/2 corpus. Computes next-word probability
distributions and generates sample text.
"""

import re
from collections import defaultdict, Counter
from nltk.tokenize import sent_tokenize
import random

random.seed(42)


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
            all_tokens.append(["<s>"] + tokens + ["</s>"])
    return all_tokens


class NGramModel:
    """A simple n-gram language model with Laplace (add-1) smoothing."""

    def __init__(self, n, sentences):
        self.n = n
        self.vocab = set()
        self.context_counts = defaultdict(Counter)  # context -> Counter of next words
        self._train(sentences)

    def _train(self, sentences):
        for tokens in sentences:
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                next_word = tokens[i + self.n - 1]
                self.context_counts[context][next_word] += 1
        self.vocab_size = len(self.vocab)

    def probability(self, context, word):
        """P(word | context) with Laplace (add-1) smoothing."""
        context = tuple(context)
        counts = self.context_counts.get(context, Counter())
        total = sum(counts.values())
        # Laplace smoothing: add 1 to every count, add vocab_size to denominator
        return (counts.get(word, 0) + 1) / (total + self.vocab_size)

    def next_word_distribution(self, context, top_k=10):
        """Returns top_k most likely next words given a context, with probabilities."""
        context = tuple(context)
        counts = self.context_counts.get(context, Counter())
        total = sum(counts.values())
        scored = []
        # Only score words that actually appeared (plus a few unseen for illustration)
        candidates = set(counts.keys())
        for word in candidates:
            prob = (counts[word] + 1) / (total + self.vocab_size)
            scored.append((word, prob))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def generate(self, seed_context, max_len=20):
        """Generate text by repeatedly sampling from the next-word distribution."""
        context = list(seed_context)
        output = list(seed_context)
        for _ in range(max_len):
            dist = self.next_word_distribution(tuple(context[-(self.n - 1):]), top_k=None if False else 50)
            if not dist:
                break
            words, probs = zip(*dist)
            # Renormalize over the top candidates for sampling
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]
            next_word = random.choices(words, weights=probs, k=1)[0]
            if next_word == "</s>":
                break
            output.append(next_word)
            context.append(next_word)
        return output


if __name__ == "__main__":
    sentences = load_corpus_tokens()
    print(f"Loaded {len(sentences)} sentences for n-gram training.\n")

    bigram = NGramModel(n=2, sentences=sentences)
    trigram = NGramModel(n=3, sentences=sentences)

    lines = ["=" * 90, "N-GRAM LANGUAGE MODELS (Laplace/add-1 smoothing)", "=" * 90]
    lines.append(f"\nVocabulary size: {bigram.vocab_size}")
    lines.append(f"Bigram contexts seen: {len(bigram.context_counts)}")
    lines.append(f"Trigram contexts seen: {len(trigram.context_counts)}")

    # --- Next-word probability distributions ---
    lines.append("\n--- BIGRAM: next-word distribution given context ---\n")
    test_contexts_bi = [("the",), ("virus",), ("function",)]
    for ctx in test_contexts_bi:
        dist = bigram.next_word_distribution(ctx, top_k=8)
        lines.append(f"P(word | '{' '.join(ctx)}'):")
        for word, prob in dist:
            lines.append(f"    {word:<15} {prob:.4f}")
        lines.append("")

    lines.append("\n--- TRIGRAM: next-word distribution given context ---\n")
    test_contexts_tri = [("the", "virus"), ("a", "new"), ("the", "function")]
    for ctx in test_contexts_tri:
        dist = trigram.next_word_distribution(ctx, top_k=8)
        lines.append(f"P(word | '{' '.join(ctx)}'):")
        if not dist:
            lines.append("    (context never seen in training data -> falls back to uniform smoothing)")
        for word, prob in dist:
            lines.append(f"    {word:<15} {prob:.4f}")
        lines.append("")

    # --- Text generation ---
    lines.append("\n--- GENERATED TEXT SAMPLES ---\n")
    for seed in [("<s>", "the"), ("<s>", "a")]:
        bigram_gen = bigram.generate(seed, max_len=15)
        trigram_gen = trigram.generate(seed, max_len=15)
        lines.append(f"Seed: {seed}")
        lines.append(f"  Bigram generation:  {' '.join(bigram_gen)}")
        lines.append(f"  Trigram generation: {' '.join(trigram_gen)}")
        lines.append("")

    # --- The key limitation ---
    lines.append("\n--- WHY THE FIXED WINDOW CAN'T CAPTURE LONG-RANGE CONTEXT ---\n")
    lines.append("A trigram model's prediction depends ONLY on the last 2 words -- it has zero")
    lines.append("memory of anything earlier in the sentence. Concretely:")
    lines.append('  Sentence: "the virus that scientists in Wuhan first identified in late 2019 spread"')
    lines.append('  When predicting the word after "spread", a trigram model only looks at')
    lines.append('  ("2019", "spread") -- it has completely forgotten "virus", "scientists", and "Wuhan"')
    lines.append("  even though those words are critical for understanding what comes next.")
    lines.append("  Increasing n (4-gram, 5-gram...) helps marginally but the context window is")
    lines.append("  still FIXED and finite, and the data sparsity problem gets rapidly worse --")
    lines.append("  most higher-order n-grams are never seen in training data at all, forcing")
    lines.append("  the model to fall back on smoothing alone. This fixed-window ceiling is")
    lines.append("  exactly what motivates sequence models (RNN/LSTM) that carry a running")
    lines.append("  hidden state across the WHOLE sequence instead of a fixed lookback window.")

    output = "\n".join(lines)
    print(output)

    with open("outputs/ngram_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/ngram_results.txt")
