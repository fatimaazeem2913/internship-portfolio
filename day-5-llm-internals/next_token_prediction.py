"""
next_token_prediction.py
---------------------------
Demonstrates next-token prediction step by step, using REAL probability
distributions computed by the trigram language model built in Day 3
(trained on our real Day 1/2 corpus). This makes autoregressive generation
concrete rather than abstract: at every step, we show the actual
distribution the "model" (our trigram LM) computed, which token got
picked, and why the whole sentence emerges one prediction at a time.

WHY A TRIGRAM MODEL FOR THIS DEMO INSTEAD OF A REAL LLM:
The mechanism being illustrated -- "predict a probability distribution
over the next token, sample one, append it, repeat" -- is IDENTICAL
between a trigram LM and GPT-4. The only difference is what produces the
distribution (frequency counts here, a 175-billion-parameter Transformer
there). Using our own real, already-trained model lets us show genuine,
verifiable numbers at every single step instead of hand-waving "the model
predicts X" without being able to show the actual math.
"""

import re
from ngram_model import NGramModel, load_corpus_tokens

import random
random.seed(7)


def tokenize_sentence(sentence):
    """Simple whitespace + lowercase tokenization, consistent with the n-gram model's training."""
    tokens = re.findall(r"[a-z']+", sentence.lower())
    return tokens


def demonstrate_step_by_step(model, seed_tokens, n_steps, top_k_display=5, backoff_model=None):
    """
    Walks through next-token prediction one step at a time, showing the
    full probability distribution the model computed at each step.

    If the trigram context was never seen in training, backs off to the
    provided backoff_model (typically a bigram model) -- exactly the kind
    of "smoothing/backoff" real statistical LMs use, and a concrete
    illustration of the data-sparsity problem discussed in Day 3.
    """
    lines = []
    context = list(seed_tokens)
    generated = list(seed_tokens)

    for step in range(n_steps):
        ctx_window = tuple(context[-(model.n - 1):])
        dist = model.next_word_distribution(ctx_window, top_k=top_k_display)
        used_model = f"trigram, context={ctx_window}"

        if not dist and backoff_model is not None:
            ctx_window_bi = tuple(context[-(backoff_model.n - 1):])
            dist = backoff_model.next_word_distribution(ctx_window_bi, top_k=top_k_display)
            used_model = f"BACKED OFF to bigram, context={ctx_window_bi} (trigram context unseen)"

        lines.append(f"\n--- STEP {step + 1} ---")
        lines.append(f"Current sequence so far: {' '.join(generated)}")
        lines.append(f"Model/context used: {used_model}")

        if not dist:
            lines.append("  (no continuation found even after backoff -- stopping)")
            break

        lines.append(f"  Top {len(dist)} candidate next tokens:")
        for word, prob in dist:
            bar = "#" * max(1, int(prob * 200))
            lines.append(f"    {word:<15} P={prob:.4f}  {bar}")

        words, probs = zip(*dist)
        total = sum(probs)
        norm_probs = [p / total for p in probs]
        chosen = random.choices(words, weights=norm_probs, k=1)[0]

        lines.append(f"  --> SAMPLED: '{chosen}'")
        if chosen == "</s>":
            lines.append("  (end-of-sequence token sampled -- generation would stop here)")
            generated.append(chosen)
            break
        generated.append(chosen)
        context.append(chosen)

    lines.append(f"\nFINAL GENERATED SEQUENCE: {' '.join(generated)}")
    return lines, generated


if __name__ == "__main__":
    all_lines = []

    def out(s=""):
        print(s)
        all_lines.append(s)

    out("=" * 90)
    out("NEXT-TOKEN PREDICTION, STEP BY STEP (using our real trigram model from Day 3)")
    out("=" * 90)

    sentences = load_corpus_tokens()
    trigram = NGramModel(n=3, sentences=sentences)
    bigram = NGramModel(n=2, sentences=sentences)

    out(f"\nModel trained on {len(sentences)} real sentences from the Day 1/2 corpus.")
    out(f"Vocabulary size: {trigram.vocab_size}")

    # --- Part A: tokenize a real sentence and show what the model does with it ---
    out("\n" + "-" * 90)
    out("PART A: Tokenizing an input sentence")
    out("-" * 90)

    input_sentence = "Scientists say the virus"
    tokens = tokenize_sentence(input_sentence)
    out(f"\nInput sentence:  \"{input_sentence}\"")
    out(f"Tokenized:       {tokens}")
    out("\n(In a real LLM this step would use BPE/WordPiece subword tokenization, as built in")
    out(" Day 1 -- here we use simple word tokenization since our trigram model was trained")
    out(" on whole words. The PRINCIPLE is identical: raw text -> a sequence of discrete units")
    out(" the model can consume.)")

    # --- Part B: step-by-step generation, fully shown ---
    out("\n" + "-" * 90)
    out("PART B: Autoregressive generation, one real prediction at a time")
    out("-" * 90)

    seed = tokens[-2:]  # use last 2 tokens as trigram context ("the", "virus")
    step_lines, final_seq = demonstrate_step_by_step(trigram, seed, n_steps=8, backoff_model=bigram)
    for l in step_lines:
        out(l)

    out("\n" + "-" * 90)
    out("THE KEY INSIGHT")
    out("-" * 90)
    out("\nNotice EXACTLY what happened at every step:")
    out("  1. The model looked at the current context (the last n-1 tokens)")
    out("  2. It produced a PROBABILITY DISTRIBUTION over every possible next token")
    out("  3. One token was sampled from that distribution")
    out("  4. That token was APPENDED to the sequence")
    out("  5. The new, longer sequence became the context for the NEXT prediction")
    out("\nThis loop -- predict distribution, sample, append, repeat -- is the ENTIRE")
    out("generation mechanism of every autoregressive language model, including GPT-4,")
    out("Claude, and every other modern LLM. The only thing that changes going from a")
    out("trigram model to a 175-billion-parameter Transformer is HOW STEP 2 computes the")
    out("distribution -- frequency counts here, billions of learned attention/FFN weights")
    out("there. The control loop wrapping that computation is identical.")

    out("\n" + "-" * 90)
    out("WHY THIS IS CALLED 'AUTOREGRESSIVE'")
    out("-" * 90)
    out("\n'Auto' = self, 'regressive' = regressing/depending on past values. The model's")
    out("own PAST OUTPUTS become its FUTURE INPUTS. Token 5 is produced by looking at")
    out("tokens 1-4 (which includes tokens the model itself already generated at steps 1-4).")
    out("This is precisely why generation is sequential (Day 3/4): you cannot compute token")
    out("5's distribution until token 4 has actually been sampled and appended, because")
    out("token 4 is part of token 5's own input.")

    with open("outputs/next_token_prediction_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))

    print("\n\nSaved to outputs/next_token_prediction_results.txt")
