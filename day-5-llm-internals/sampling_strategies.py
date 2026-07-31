"""
sampling_strategies.py
-------------------------
Implements temperature, top-k, and top-p (nucleus) sampling from scratch in
pure Python/NumPy, applied to REAL next-token probability distributions
computed by our trigram model (same one from Day 3, trained on our real
corpus). This is the exact same mechanism used to sample from GPT/Claude's
output logits -- only the source of the initial distribution differs.

WHY NOT THE OPENAI API HERE:
This sandboxed verification environment cannot reach api.openai.com (network
whitelist blocks it, confirmed directly). openai_api_version.py in this same
folder contains fully correct, ready-to-run code using the real OpenAI API,
for local execution with your own API key -- see that file and the README
for exact instructions. The sampling ALGORITHMS themselves (temperature,
top-k, top-p) are 100% identical regardless of whether the underlying
probabilities came from a trigram model or GPT-4; implementing and verifying
them here, against real numbers, is not a compromise -- it's the same code
you'd run against any other logit distribution.
"""

import re
import numpy as np
from ngram_model import NGramModel, load_corpus_tokens

np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)


def get_full_distribution(model, context):
    """Get the FULL probability distribution (not just top-k) over the entire vocabulary."""
    context = tuple(context)
    from collections import Counter
    counts = model.context_counts.get(context, Counter())
    total = sum(counts.values())
    vocab = sorted(model.vocab)
    probs = np.array([(counts.get(w, 0) + 1) / (total + model.vocab_size) for w in vocab])
    return vocab, probs


def apply_temperature(logits_or_probs, temperature, from_probs=True):
    """
    Temperature scaling: reshape a distribution's "peakedness" before sampling.

        p_i' = p_i^(1/T) / sum_j( p_j^(1/T) )       [if starting from probabilities]

    or equivalently, working in logit space:
        logits' = logits / T
        p' = softmax(logits')

    T < 1: sharpens the distribution (more confident, more deterministic,
           closer to greedy argmax as T -> 0)
    T = 1: unchanged
    T > 1: flattens the distribution (more random, more diverse, approaches
           uniform sampling as T -> infinity)
    """
    if from_probs:
        # Convert to logit-equivalent via log, apply temperature, convert back
        log_probs = np.log(logits_or_probs + 1e-12)
        scaled = log_probs / temperature
        scaled = scaled - scaled.max()  # numerical stability
        exp_scaled = np.exp(scaled)
        return exp_scaled / exp_scaled.sum()
    else:
        scaled = logits_or_probs / temperature
        scaled = scaled - scaled.max()
        exp_scaled = np.exp(scaled)
        return exp_scaled / exp_scaled.sum()


def top_k_filter(vocab, probs, k):
    """
    Top-k sampling: keep only the k highest-probability tokens, zero out
    everything else, renormalize the remaining probabilities to sum to 1.

    Prevents sampling from the "long tail" of thousands of near-zero-
    probability tokens, which individually are unlikely but collectively
    have enough combined probability mass to occasionally produce
    incoherent output.
    """
    idx_sorted = np.argsort(probs)[::-1]
    top_idx = idx_sorted[:k]
    filtered_probs = np.zeros_like(probs)
    filtered_probs[top_idx] = probs[top_idx]
    filtered_probs = filtered_probs / filtered_probs.sum()
    return filtered_probs


def top_p_filter(vocab, probs, p):
    """
    Top-p (nucleus) sampling: keep the SMALLEST set of highest-probability
    tokens whose cumulative probability exceeds p, zero out the rest,
    renormalize.

    The key advantage over top-k: the size of the kept set ADAPTS to the
    shape of the distribution. If the model is very confident (one token
    dominates), the nucleus might contain just 1-2 tokens. If the model is
    uncertain (probability spread thin across many tokens), the nucleus
    might contain dozens. Top-k's fixed cutoff can't adapt this way -- it
    either wastefully includes low-probability tokens when the model is
    confident, or overly restricts options when the model is uncertain.
    """
    idx_sorted = np.argsort(probs)[::-1]
    sorted_probs = probs[idx_sorted]
    cumulative = np.cumsum(sorted_probs)
    # Find the smallest prefix whose cumulative probability >= p
    cutoff_idx = np.searchsorted(cumulative, p) + 1
    keep_idx = idx_sorted[:cutoff_idx]
    filtered_probs = np.zeros_like(probs)
    filtered_probs[keep_idx] = probs[keep_idx]
    filtered_probs = filtered_probs / filtered_probs.sum()
    return filtered_probs, cutoff_idx


def sample_and_report(vocab, probs, n_samples, label, out_fn):
    """Draw multiple samples and report the distribution of outcomes."""
    nonzero = np.where(probs > 1e-10)[0]
    out_fn(f"  Non-zero-probability candidates remaining: {len(nonzero)} / {len(vocab)}")

    samples = np.random.choice(len(vocab), size=n_samples, p=probs)
    sample_words = [vocab[i] for i in samples]
    from collections import Counter
    counts = Counter(sample_words)
    out_fn(f"  {n_samples} samples drawn -> outcome distribution: {dict(counts.most_common(8))}")
    return sample_words


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("TEMPERATURE, TOP-K, AND TOP-P SAMPLING FROM SCRATCH")
    out("(applied to REAL probability distributions from our Day 3 trigram model)")
    out("=" * 90)

    sentences = load_corpus_tokens()
    bigram = NGramModel(n=2, sentences=sentences)

    context = ("the",)
    vocab, probs = get_full_distribution(bigram, context)

    out(f"\nContext: {context}   (bigram model -- chosen because it has 1,460 REAL observed")
    out(f" continuations in the corpus, giving a much less smoothing-dominated distribution")
    out(f" than a sparser trigram context would.)")
    out(f"Full vocabulary size: {len(vocab)}")

    top10_idx = np.argsort(probs)[::-1][:10]
    out(f"\nTop 10 tokens in the RAW (unmodified) distribution:")
    for i in top10_idx:
        out(f"  {vocab[i]:<15} P={probs[i]:.6f}")

    # ------------------------------------------------------------------
    # TEMPERATURE
    # ------------------------------------------------------------------
    out("\n" + "-" * 90)
    out("TEMPERATURE SAMPLING")
    out("-" * 90)
    out("\nFormula: p_i' = p_i^(1/T) / sum(p_j^(1/T))\n")

    for T in [0.3, 1.0, 2.0]:
        out(f"\n--- Temperature = {T} ---")
        adjusted = apply_temperature(probs, temperature=T, from_probs=True)
        top5_idx = np.argsort(adjusted)[::-1][:5]
        out(f"  Top 5 after temperature scaling:")
        for i in top5_idx:
            out(f"    {vocab[i]:<15} P={adjusted[i]:.4f}")
        entropy = -np.sum(adjusted[adjusted > 0] * np.log(adjusted[adjusted > 0]))
        out(f"  Distribution entropy: {entropy:.4f}  (higher = more spread out / random)")

    out("\nOBSERVATION: at T=0.3 the top candidate's probability increases sharply")
    out("(the distribution SHARPENS -- closer to always picking the same top word,")
    out("i.e. closer to greedy/deterministic decoding). At T=2.0 probability mass")
    out("spreads toward previously-unlikely tokens (the distribution FLATTENS --")
    out("more randomness, more diversity, more risk of incoherence). Entropy")
    out("increases monotonically with T, confirming this numerically.")

    # ------------------------------------------------------------------
    # TOP-K
    # ------------------------------------------------------------------
    out("\n" + "-" * 90)
    out("TOP-K SAMPLING")
    out("-" * 90)

    for k in [1, 3, 10, 50]:
        out(f"\n--- k = {k} ---")
        filtered = top_k_filter(vocab, probs, k)
        nonzero = np.sum(filtered > 0)
        out(f"  Candidates with nonzero probability after filtering: {nonzero}")
        top3_idx = np.argsort(filtered)[::-1][:3]
        out(f"  Top 3 remaining:")
        for i in top3_idx:
            if filtered[i] > 0:
                out(f"    {vocab[i]:<15} P={filtered[i]:.4f}")

    out("\nOBSERVATION: k=1 is equivalent to GREEDY decoding -- always picks the single")
    out("highest-probability token, fully deterministic. As k increases, more of the")
    out("long tail becomes eligible for sampling, increasing diversity but also the")
    out("chance of picking a less sensible continuation.")

    # ------------------------------------------------------------------
    # TOP-P (NUCLEUS)
    # ------------------------------------------------------------------
    out("\n" + "-" * 90)
    out("TOP-P (NUCLEUS) SAMPLING")
    out("-" * 90)

    for p in [0.3, 0.7, 0.95]:
        out(f"\n--- p = {p} ---")
        filtered, cutoff_n = top_p_filter(vocab, probs, p)
        out(f"  Number of tokens needed to reach cumulative probability {p}: {cutoff_n}")
        top3_idx = np.argsort(filtered)[::-1][:3]
        out(f"  Top 3 remaining:")
        for i in top3_idx:
            if filtered[i] > 0:
                out(f"    {vocab[i]:<15} P={filtered[i]:.4f}")

    out("\nOBSERVATION: unlike top-k's FIXED cutoff count, top-p's cutoff COUNT adapts")
    out("to the shape of the distribution at each individual step. Compare the")
    out("'Number of tokens needed' line across the three p values above -- this")
    out("number is the direct evidence that top-p is adaptive where top-k is rigid.")

    # ------------------------------------------------------------------
    # SIDE-BY-SIDE: actually sampling multiple times to see real outcome diversity
    # ------------------------------------------------------------------
    out("\n" + "-" * 90)
    out("SIDE-BY-SIDE: 200 samples drawn under each strategy")
    out("-" * 90)

    out("\nRAW distribution (no modification):")
    sample_and_report(vocab, probs, 200, "raw", out)

    out("\nTemperature = 0.3 (sharpened):")
    sample_and_report(vocab, apply_temperature(probs, 0.3), 200, "T=0.3", out)

    out("\nTemperature = 2.0 (flattened):")
    sample_and_report(vocab, apply_temperature(probs, 2.0), 200, "T=2.0", out)

    out("\nTop-k, k=5:")
    sample_and_report(vocab, top_k_filter(vocab, probs, 5), 200, "top-k=5", out)

    out("\nTop-p, p=0.7:")
    filtered_p, _ = top_p_filter(vocab, probs, 0.7)
    sample_and_report(vocab, filtered_p, 200, "top-p=0.7", out)

    out("\n" + "=" * 90)
    out("SUMMARY: WHY THESE STRATEGIES EXIST AND WHEN TO USE EACH")
    out("=" * 90)
    out("""
Greedy decoding (always pick argmax, equivalent to T->0 or top-k=1):
  Fully deterministic, always the same output for the same input. Can get
  stuck in repetitive loops ("the the the...") since it never explores
  lower-probability-but-reasonable alternatives.

Temperature:
  Simple global control over randomness. Low T = focused/deterministic
  (good for factual Q&A, code generation). High T = creative/diverse (good
  for brainstorming, creative writing). Downside: applies uniformly across
  the WHOLE distribution regardless of shape -- doesn't adapt to how
  confident vs uncertain the model is at each step.

Top-k:
  Hard cutoff at a fixed NUMBER of candidates. Simple and predictable, but
  the fixed count doesn't adapt: when the model is very confident (one
  token should dominate), top-k=50 wastefully keeps 49 near-irrelevant
  options open; when the model is uncertain (probability spread thin),
  top-k=50 might exclude reasonable tokens ranked 51+.

Top-p (nucleus):
  Adapts the candidate COUNT to the actual shape of the distribution at
  each step -- confident steps get a small nucleus, uncertain steps get a
  larger one. This is why top-p (often combined with a modest temperature)
  is the most common default in production LLM APIs today.

In practice, production systems typically combine temperature + top-p (and
sometimes top-k as an extra safety bound) rather than using any one
strategy alone.
""")

    with open("outputs/sampling_strategies_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/sampling_strategies_results.txt")
