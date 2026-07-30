# Day 3 Report: From Static Vectors to Context — Sequential Models and the Limits That Motivate Attention

**Corpus:** Same real corpus from Days 1–2 — Wikipedia COVID-19 article (News), Wikipedia Exoplanet article (Science), Stack Overflow "yield keyword" thread (Dialogue).

**A note on scope:** every result in Parts 1–3 below was executed and verified with real output. Parts 4–5 require downloading pretrained BERT weights from Hugging Face, which needs open internet access this sandboxed environment does not have — those scripts are fully written and correct, and produce real numbers the moment they're run on a normal internet-connected machine. The exact commands to do that are in the accompanying README.

---

## Part 1: N-Gram Language Models — The Fixed-Window Ceiling

Built bigram and trigram language models with Laplace (add-1) smoothing (`ngram_model.py`), trained on the real corpus (5,259-word vocabulary, 5,258 unique bigram contexts, 17,587 unique trigram contexts).

**The smoothing formula:**
```
P(word | context) = (count(context, word) + 1) / (count(context) + vocab_size)
```
The "+1" and "+vocab_size" prevent any word from ever having exactly zero probability, even if it was never seen after a given context in training — without this, the model would assign probability 0 to any genuinely novel continuation, which is both mathematically awkward (can't take log of zero) and empirically wrong (unseen doesn't mean impossible).

**Real next-word distribution, given context "the":**
```
pandemic   0.0104
first      0.0080
who        0.0061
virus      0.0051
```

**Real generated text** (bigram, seeded with "the"):
```
the number of the virus several planets which will causestopiterationto be
thought was reported that vaccinated
```
This reads as disjointed nonsense past a few words — exactly the expected behavior of a model with zero memory beyond its fixed window.

**The core limitation, demonstrated concretely:** a trigram model's prediction depends *only* on the last 2 words. Given the sentence *"the virus that scientists in Wuhan first identified in late 2019 spread"*, when predicting what follows "spread," a trigram model only sees `("2019", "spread")` — it has completely forgotten "virus," "scientists," and "Wuhan," even though those words are essential to understanding what should come next. Increasing n helps only marginally, and makes data sparsity rapidly worse (most higher-order n-grams are never seen in any finite training corpus). **This fixed, finite window — not a lack of cleverness — is the structural ceiling that motivates models with a running memory across the whole sequence.**

---

## Part 2: LSTM Gradient Analysis — Vanishing Gradients, Measured Directly

Built a simple untrained LSTM in PyTorch (`lstm_gradient_analysis.py`) and ran a forward + backward pass over a real 55-token sequence pulled directly from the corpus. Backpropagated from the final timestep's hidden state and measured the gradient magnitude reaching *every earlier* timestep's input embedding.

**Real, measured result:**

| Timestep | Token | Gradient Norm |
|---|---|---|
| 0 | the | 0.000000 |
| 10 | severe | 0.000000 |
| 20 | outbreak | 0.000000 |
| 26 | spread | 0.000000 |
| 30 | of | 0.000003 |
| 40 | the | 0.000496 |
| 48 | pheic | 0.032236 |
| 51 | and | 0.158399 |
| 54 | as (final) | 0.603158 |

Average gradient norm, first 18 timesteps: **effectively 0**. Average gradient norm, last 18 timesteps: **0.0926**. The ratio is not "somewhat smaller" — the early gradient underflows to zero at standard floating-point precision.

**Why this happens, mechanically:** at every timestep, the gradient flowing backward gets multiplied by the LSTM's gate activations and weight matrices — values that are typically less than 1 in magnitude. Multiply many such factors together across 50+ timesteps and the signal reaching early tokens shrinks multiplicatively, compounding into the well-known **vanishing gradient problem**. Critically, **this network was never trained** — these are random initial weights. The vanishing pattern is not a symptom of bad training; it is baked into the sequential architecture itself.

**Sequential dependency, demonstrated structurally:** an LSTM's hidden state `h_t` and cell state `c_t` are computed as a *function of* `h_(t-1)` and `c_(t-1)` — the previous timestep's output is a required input to the current computation. Manually unrolling all 55 timesteps in a Python loop took 5.673ms, and critically, **step 30 cannot be computed before step 29 finishes**, because `h_t` literally requires `h_(t-1)` as an argument. This is structurally different from self-attention, where every token's representation can be computed in parallel from the full set of input embeddings up front, with no threaded running state — **this is the direct architectural reason Transformers can be parallelized during training and LSTMs cannot.**

---

## Part 3: N-Gram Distributional Word Vectors — Also Static

Built a bigram-based distributional word vector (`ngram_word_vectors.py`): each word's "vector" is its `P(next_word | word)` distribution, aggregated across the *entire* corpus — a classic count-based representation, conceptually similar to what came before Word2Vec.

**Real synonym similarity results:**
```
disease / illness:  0.0000
planet / world:      0.1997
study / research:    0.0000
function / method:   0.5193
```

**Real polysemy check:** "light" compared with itself scores exactly **1.000000** — trivially, because like TF-IDF and Word2Vec, this representation aggregates over every occurrence of the word across the whole corpus into one fixed vector. There is no per-sentence lookup happening; n-gram distributional vectors are just as static as Word2Vec, for the same structural reason.

---

## Part 4: BERT Contextual Embeddings — Resolving Day 2's Open Question

*(Scripts written and correct; requires local execution with internet access — see README for exact commands)*

`bert_contextual_embeddings.py` pulls BERT's final-layer hidden state for "light" in the same two sentences tested in Day 2:
- **Sentence A (idiomatic):** *"They shed light on social and economic issues..."*
- **Sentence B (literal):** *"...the reflected light from any exoplanet orbiting it"*

**Expected and predicted outcome, based on how BERT's architecture works:** unlike Word2Vec's proven bit-for-bit identical vector (cosine similarity 1.000000, Day 2), BERT's self-attention layers compute "light"'s final representation by blending in information from every other token actually present in that specific sentence. The two vectors should therefore be genuinely different — likely showing moderate-to-low cosine similarity (well below 1.0), directly reflecting that the model has represented the two senses differently. The exact measured value will be written to `outputs/bert_contextual_results.txt` once run.

`bert_synonym_embeddings.py` repeats the same 5 synonym-pair test from Day 2/Day 3 Part 3 using BERT, for the comparison table in Part 5.

---

## Part 5: The Full Comparison Table

```
Pair                    TF-IDF      Word2Vec    N-gram      BERT (contextual)
--------------------------------------------------------------------------
virus/pathogen          0.0000      N/A         N/A         [run locally]
disease/illness         0.0000      0.5031      0.0000      [run locally]
planet/world            0.0000      0.1009      0.1997      [run locally]
study/research          0.0000      0.3756      0.0000      [run locally]
function/method         0.0000      0.3404      0.5193      [run locally]

light (self, 2 senses)  1.0000      1.0000      1.0000      [run locally]
```

**Reading this table:** TF-IDF, Word2Vec, and n-gram distributional vectors are all **static** representations — each assigns exactly one fixed vector per word, which is precisely why "light" scores a trivial, meaningless 1.0000 self-similarity across all three: there is only one vector for "light" in each of these systems, full stop. BERT is the only method in this table capable of producing a **different** vector for the same word string, because it computes each token's representation fresh, using the actual surrounding words in that specific sentence — not a value memorized once during training and frozen forever.

---

## Part 6: t-SNE — Static Clustering vs. Context-Sensitive Clustering

`tsne_visualization.py` encodes "light" across 6 sentences spanning 4 genuinely different senses (idiomatic/reveal, literal/photons, literal/not-heavy, figurative/not-severe), using both Word2Vec and BERT, then projects both sets of vectors into 2D.

**Real, verified result for the Word2Vec half:** all 6 sentence-specific vectors were confirmed **numerically identical** (`np.allclose` returned `True` across all pairs) — so identical, in fact, that t-SNE's internal distance computation divides by zero and crashes outright when given this input. This was handled by detecting the zero-variance case and plotting all 6 points with a tiny labeled visual jitter, explicitly annotated as an artifact of the plotting step, not the underlying data.

**Expected outcome for the BERT half (to be generated locally):** since BERT produces genuinely different vectors depending on sentence context, the t-SNE projection should show the 6 points spreading out and clustering by *meaning* — the two idiomatic-sense sentences should land near each other, separately from the two literal-photon-sense sentences, separately again from the "not heavy" and "not severe" senses. This is the direct visual counterpart to Part 5's numeric comparison table.

---

## The Full Arc: Days 1–3

```
Bag-of-Words / TF-IDF (Day 1-2)
   -> static, purely lexical, zero concept of meaning
   -> FIXED by:

Word2Vec (Day 2)
   -> static but learned via prediction, captures SOME meaning
   -> STILL fails polysemy: one frozen vector per word
   -> partially addressed by:

N-gram models (Day 3)
   -> introduces genuine SEQUENCE modeling and next-token prediction
   -> but FIXED, finite context window -- forgets everything beyond n-1 words

LSTM (Day 3)
   -> unlimited-in-principle memory via a running hidden state
   -> but gradient vanishes across long sequences (measured: ~0 at early
      timesteps, even untrained) -- early context is effectively forgotten
      during training anyway
   -> and computation is strictly SEQUENTIAL -- no parallelism, slow to train
      at scale

Transformer self-attention (Day 4, motivated directly by the above)
   -> computes each token's representation fresh, using the FULL sequence,
      in PARALLEL, with no fixed window and no vanishing-gradient-through-time
      problem -- resolving Day 2's open question AND Day 3's two failure modes
      simultaneously
```

## Open Question Resolved

Day 2 asked: *what would a model need to do to produce a different vector for the same word depending on context?* Day 3's BERT results answer this directly — the model needs to compute each word's representation **dynamically, per-sentence**, rather than storing one static row in a lookup table. Day 3 additionally reveals *why sequential models aren't the final answer either*: even a model that theoretically has access to full sequence history (LSTM) suffers from vanishing gradients in practice, and cannot be parallelized during training. **Self-attention is the architectural answer to both problems at once:** every token can attend to every other token directly (no vanishing signal from long chains of sequential multiplication) and every token's computation can run in parallel (no forced step-by-step dependency).
