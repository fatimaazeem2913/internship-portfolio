# Day 4 Report: Transformer Architecture & Attention Mechanisms

**Objective:** Build a solid technical understanding of how Transformer models process sequences, enabling architecture-level reasoning about LLM behaviour.

---

## Part 1: "Attention Is All You Need" (Vaswani et al., 2017)

### The problem it solved

By 2017, the state of the art for sequence transduction (translation, summarization) was the encoder-decoder RNN/LSTM with attention. These models worked, but had two structural problems that Days 1–3 of this internship measured directly:

**Problem 1 — Sequential computation blocks parallelism.** An RNN computes hidden state `h_t` as a function of `h_{t-1}`. Timestep 30 cannot begin until timestep 29 finishes. On a GPU with thousands of cores, this means the hardware sits mostly idle while the model plods through one position at a time. Training time scales linearly with sequence length in a way that cannot be parallelized away.

*(Measured directly in Day 3: manually unrolling a 55-token LSTM required 55 strictly ordered steps, each depending on the previous.)*

**Problem 2 — Long-range dependencies degrade.** Information from token 1 reaching token 50 must pass through 49 intermediate transformations. Gradients flowing backward through that chain get multiplied by sub-1 values at every step and vanish.

*(Measured directly in Day 3: gradient magnitude at the earliest timesteps of a 55-token sequence was 0.000000 while the final timestep received 0.603158 — in an untrained network, confirming the problem is architectural.)*

The paper frames this in terms of **path length**: how many operations must information traverse to get from position i to position j? For an RNN it's O(n) — linear in the distance. Short paths make learning long-range dependencies easier; long paths make it harder.

### The core innovation

Replace recurrence entirely with self-attention. The paper's title is a literal claim: you don't need the recurrent connections at all.

In self-attention, **every position attends to every other position in a single operation.** The path length between any two tokens is O(1) — constant, regardless of how far apart they are. Token 1 and token 500 are exactly as "close" to each other computationally as token 1 and token 2.

This one change resolves both problems at once:
- **Parallelism:** every token's Q, K, and V vectors are computed from its own input independently. All positions can be processed simultaneously as a single matrix multiplication. There is no running state to thread through the sequence.
- **Gradient flow:** the gradient from the output back to any input token traverses roughly the same number of operations regardless of position, so no position is systematically disadvantaged.

### Complexity comparison (paper Table 1)

| Layer type | Complexity per layer | Sequential operations | Max path length |
|---|---|---|---|
| Recurrent (RNN/LSTM) | O(n · d²) | **O(n)** | **O(n)** |
| Convolutional | O(k · n · d²) | O(1) | O(log_k(n)) |
| Self-Attention | **O(n² · d)** | **O(1)** | **O(1)** |

Read the tradeoff carefully. Self-attention's *compute* cost per layer is worse than recurrence for long sequences — it's quadratic in sequence length `n`, because every token attends to every other token (n × n attention matrix). But its *sequential operations* count is O(1) and its *path length* is O(1).

The bet the paper makes: **quadratic-but-parallel beats linear-but-sequential on modern hardware.** A GPU can execute a huge n² matrix multiplication in one pass; it cannot parallelize n sequential steps at all. This bet turned out to be overwhelmingly correct, and the resulting quadratic cost is precisely why "context window" became the central engineering constraint of the LLM era.

### Why self-attention replaced RNNs — the honest summary

Not because it's cheaper in raw FLOPs (it isn't, for long sequences), but because:
1. **It parallelizes.** This is the decisive practical factor — it made training on internet-scale data economically feasible.
2. **Path length is constant.** Long-range dependencies become as learnable as short-range ones.
3. **Attention weights are interpretable.** You can inspect which tokens attended to which — a genuine practical bonus over an RNN's opaque hidden state.

---

## Part 2: The Five Components

### 2.1 Multi-Head Self-Attention

**Scaled dot-product attention** (paper Section 3.2.1):

```
Attention(Q, K, V) = softmax( (Q Kᵀ) / √d_k ) V
```

Every token produces three vectors via learned projections:
- **Query (Q)** — "what information am I looking for?"
- **Key (K)** — "what information do I contain?"
- **Value (V)** — "what do I actually contribute if attended to?"

The mechanism, step by step:
1. `Q Kᵀ` computes a compatibility score between every query and every key — an n × n matrix.
2. Divide by `√d_k` (the scaling step — see below).
3. Softmax over the key axis, so each token's attention weights form a probability distribution summing to 1.
4. Multiply by V: each token's output is a weighted average of all tokens' value vectors.

**Why the √d_k scaling — verified empirically in `scaled_dot_product_attention.py`:**

Dot products over `d_k` dimensions have variance proportional to `d_k`. As `d_k` grows, the raw scores grow in magnitude, pushing softmax toward a one-hot distribution where one weight ≈ 1.0 and the rest ≈ 0. Softmax's gradient in that saturated regime is near zero, so learning stalls.

Real measured demonstration:

| d_k | Raw score std dev | Max weight WITHOUT scaling | Max weight WITH scaling |
|---|---|---|---|
| 4 | 1.5326 | — | — |
| 64 | 6.6735 | **0.9996** | 0.5321 |
| 512 | 24.8373 | **1.0000** | 0.6265 |

At `d_k = 512`, unscaled softmax produces a max weight of 1.0000 — completely saturated, zero gradient. Scaling holds it at 0.6265, a healthy trainable range. **This is the entire reason the mechanism is called *scaled* dot-product attention.**

**Multi-head attention** runs `h` attention operations in parallel over `d_model / h` dimensions each, then concatenates and projects:

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O
    where head_i = Attention(Q W_Qⁱ, K W_Kⁱ, V W_Vⁱ)
```

**Why multiple heads:** a single softmax distribution can only express one "kind" of relationship at a time — if a head attends strongly to the syntactic subject, it cannot simultaneously attend strongly to a semantically related word elsewhere. With 8–16 heads, different heads specialize (syntax, coreference, positional proximity), and their concatenated outputs give a far richer representation. Verified in `scaled_dot_product_attention.py`: 8 heads produced 8 genuinely different attention distributions over identical input.

### 2.2 Position-wise Feed-Forward Network

```
FFN(x) = activation(x W₁ + b₁) W₂ + b₂
```

Applied **independently and identically to every token position** — same weights, no cross-token mixing. Verified in `transformer_block.py`: running the FFN on a single token in isolation produced a bit-for-bit identical result to running it as part of the full sequence.

This is a crucial division of labour:
- **Attention** is where tokens communicate with each other.
- **FFN** is where each token processes its own accumulated information.

The hidden dimension is conventionally **4× d_model** (768 → 3072 → 768 in GPT-2 Small). The original paper used ReLU; GPT and BERT use GELU, which is smooth and permits small negative values, empirically training better at depth.

**Parameter accounting (verified numerically):**

| Component | GPT-2 Small block (d_model=768) | Share |
|---|---|---|
| Attention (W_q, W_k, W_v, W_o) | 2,359,296 | 33.3% |
| Feed-Forward Network | 4,722,432 | **66.7%** |
| LayerNorm parameters | 3,072 | 0.0% |
| **Total per block** | **7,084,800** | 100% |

**Counterintuitive but important:** the FFN holds two-thirds of every block's parameters, despite attention receiving nearly all the conceptual discussion. LayerNorm's contribution is negligible.

### 2.3 Layer Normalization

```
LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β
```
where μ and σ² are computed **across the feature dimension for each token independently**.

**Why LayerNorm and not BatchNorm:** BatchNorm's statistics depend on other examples in the batch. That's problematic for variable-length sequences and breaks entirely at inference with batch size 1. LayerNorm computes statistics from only the single token being normalized, so it behaves identically at training and inference regardless of batch composition.

γ (scale) and β (shift) are learned, so the network can undo the normalization if useful — normalization never costs expressive power.

Verified in `transformer_block.py`: a token with values ranging from −5 to 100 was normalized using only its own statistics, producing mean ≈ 0 and std ≈ 1, without reference to any other token.

### 2.4 Residual Connections

```
output = x + Sublayer(x)
```

**Why this matters — the derivative:**
```
d(x + F(x))/dx = 1 + dF(x)/dx
```

That `+1` is the whole point. Even if the sublayer's local gradient `dF/dx` is tiny, the gradient still flows backward at full strength through the identity path. Stacking 96 layers is only feasible because every residual connection provides an unobstructed gradient highway.

**Measured demonstration** (local derivative dF/dx = 0.7):

| Layers | Gradient without residual | Gradient with residual |
|---|---|---|
| 10 | 2.825e-02 | 2.016e+02 |
| 50 | 1.798e-08 | 3.330e+11 |
| 96 | **1.347e-15** | 1.328e+22 |

Without residuals the gradient vanishes exponentially — the exact same mathematics that produced Day 3's LSTM measurement of 0.000000. With residuals it survives, but *grows*, which is precisely why residuals and LayerNorm must be used as a pair: **residuals guarantee gradient flow, LayerNorm keeps the resulting magnitudes numerically stable.**

**Pre-Norm vs Post-Norm:**
- **Post-Norm** (original paper): `LayerNorm(x + Sublayer(x))`
- **Pre-Norm** (GPT-2 onward, modern standard): `x + Sublayer(LayerNorm(x))`

Pre-Norm keeps the residual stream itself un-normalized, so the gradient highway is never interrupted by a normalization step. This trains more stably at extreme depth, which is why essentially all modern LLMs use it. Verified in `transformer_block.py`: Post-Norm output had std ≈ 0.999995 (tightly normalized, since LayerNorm is the final op); Pre-Norm output had std ≈ 0.951478 (retaining the residual stream's natural scale).

### 2.5 Positional Encoding

**The problem:** self-attention is **permutation-invariant.** Shuffle the input tokens and you get the same outputs in shuffled order — the mechanism has no inherent notion of order. An RNN gets order for free by processing sequentially; a Transformer does not.

**Verified directly in `positional_encoding.py`:**
```
WITHOUT positional encoding:
  Output for token at original position: [ 1.5041 -0.2272  0.3093 ...]
  Same token after shuffling input:      [ 1.5041 -0.2272  0.3093 ...]
  IDENTICAL: True     <- cannot distinguish "dog bites man" from "man bites dog"

WITH positional encoding added:
  Output for token at original position: [ 2.3910 -0.6983  0.5029 ...]
  Same token after shuffling input:      [ 1.5617  0.6776  0.3249 ...]
  IDENTICAL: False    <- order information successfully injected
```

**The sinusoidal formula:**
```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

**Four properties, all verified numerically:**

1. **Bounded in [−1, 1].** Measured min −1.000000, max 1.000000 over a 512×128 encoding. This matters because PE is *added* to word embeddings — raw position integers (0…511) would swamp the semantic content entirely.

2. **Unique per position.** 512 distinct encoding vectors across 512 positions.

3. **Relative positions are linear transformations.** For each frequency band, shifting position by offset k is exactly a 2D rotation:
   ```
   [sin(ω(pos+k))]   [ cos(ωk)  sin(ωk)] [sin(ω·pos)]
   [cos(ω(pos+k))] = [−sin(ωk)  cos(ωk)] [cos(ω·pos)]
   ```
   Verified across the first four frequency bands at pos=3, k=5 — all matched to within 1e-9. **Why this matters:** because relative offsets are linear, the model can learn a single weight matrix meaning "attend to whatever is 3 positions back" and apply it at any absolute position.

4. **Similarity decays with distance.** Measured cosine similarity between PE(0) and PE(k):

| k | 0 | 1 | 5 | 20 | 100 | 200 |
|---|---|---|---|---|---|---|
| cos sim | 1.0000 | 0.9702 | 0.7373 | 0.6083 | 0.4772 | 0.3057 |

This gives the model a continuous, usable notion of "how far apart are these tokens" without ever explicitly computing a distance. Lower dimensions oscillate fast (fine-grained local position); higher dimensions oscillate slowly (coarse long-range position) — together forming a unique continuous signature per position, structurally analogous to a binary counter.

---

## Part 3: Critical Terminology Distinctions

### Tokens vs Words

| | Words | Tokens |
|---|---|---|
| Definition | Linguistic units separated by whitespace | Units the model actually processes, from a fixed vocabulary |
| Example | "unbelievable" = 1 word | "unbelievable" may be 1 token, or `un` + `believ` + `able` |
| Vocabulary | Unbounded (new words constantly) | Fixed (e.g. 50,257 for GPT-2) |
| Unknown inputs | N/A | Broken into known subword pieces — no `[UNK]` |

Practical consequences: token counts drive API pricing and context limits, not word counts. English averages roughly 1.3 tokens per word; code and non-English text are often far less efficient. A model literally cannot "see" letters inside a token — which is why LLMs historically struggled to count letters in a word or reverse a string.

### Training vs Inference

| | Training | Inference |
|---|---|---|
| Goal | Adjust weights to reduce loss | Produce output with weights frozen |
| Weights | Updated every step via backprop | Never change |
| Direction | Forward pass, then backward pass | Forward pass only |
| Parallelism over sequence | **Full** — all positions computed at once, causal mask prevents cheating | **Limited** — generation is inherently one token at a time |
| Cost | Enormous, one-time | Small per request, but repeated forever |

**The critical asymmetry:** during training, a decoder-only model computes the prediction for *every* position in a sequence simultaneously in one forward pass, because the correct answers are already known and the causal mask prevents each position seeing the future. During generation, position n+1's input *is* position n's output, so tokens must be produced sequentially. This is why generating 1,000 tokens takes 1,000 forward passes, and it's the fundamental reason LLM inference latency scales with output length.

### Context Window vs Memory

| | Context window | Memory (as usually meant) |
|---|---|---|
| What it is | Max tokens the model can attend over in one forward pass | Persistent information across separate conversations |
| Nature | A hard architectural constraint | Not an architectural feature at all |
| Mechanism | Attention over the tokens present | Implemented externally: retrieval, summarization, databases |
| Persistence | Vanishes completely when the request ends | Deliberately stored outside the model |

**The essential point: a Transformer has no memory.** It has a context window. Every request is processed from scratch with no state carried over. What feels like memory in a chat interface is the application re-sending prior conversation turns as part of the input each time. Once tokens fall outside the window, they are simply gone — not "forgotten," never stored in the first place.

The window is finite because attention is O(n²): doubling context quadruples attention compute and memory. This is why long-context capability is a genuine engineering achievement rather than a config change.

### Embeddings vs Tokens

| | Tokens | Embeddings |
|---|---|---|
| Type | Discrete integers (vocabulary indices) | Continuous vectors of floats |
| Example | `["The", "cat"]` → `[464, 3797]` | `464` → `[0.021, −0.318, ..., 0.107]` (768 floats) |
| Carries meaning? | No — arbitrary IDs, ID 464 isn't "more" than 463 | Yes — geometric position encodes semantics |
| Learned? | No — vocabulary is fixed after tokenizer training | Yes — the embedding table is trained parameters |

The pipeline: **text → tokens (integers) → embeddings (vectors) → +positional encoding → transformer layers.** Tokens are addresses; embeddings are the contents at those addresses. A token ID is a pointer with no intrinsic meaning; its embedding is a learned point in high-dimensional space whose geometry encodes meaning.

Critically — and this is the entire Day 2 → Day 3 arc — the initial embedding is *static* (one fixed vector per token ID, exactly like Word2Vec), but after passing through the transformer layers, each token's representation becomes *contextual*. Day 3 measured this: Word2Vec's "light" had cosine similarity 1.000000 across two senses; BERT's contextual representation of the same word had 0.3810.

---

## Part 4: GPT-2 and Decoder-Only Architecture (via nanoGPT)

Andrej Karpathy's nanoGPT is a ~300-line reimplementation of GPT-2 that makes the architecture unusually legible. Studying it clarifies exactly how a decoder-only model differs from the original 2017 encoder-decoder design.

### The original 2017 architecture: encoder-decoder

Designed for **sequence-to-sequence** tasks (machine translation), with three distinct attention mechanisms:

1. **Encoder self-attention** — bidirectional. Every source token attends to every other source token, including future ones. Legitimate, because the entire source sentence is available up front.
2. **Decoder masked self-attention** — causal. Target tokens attend only to earlier target tokens, because generation is left-to-right.
3. **Cross-attention** — decoder queries attend to encoder keys/values. This is the bridge: it's how the target sequence conditions on the source.

### GPT: decoder-only

GPT removes the encoder and cross-attention entirely, keeping only masked self-attention. The task is reframed: instead of "translate this source sequence into this target sequence," everything becomes **"predict the next token given all previous tokens."** Translation, summarization, and Q&A all become special cases of next-token prediction on appropriately formatted text.

### Comparison

| | Original Transformer (2017) | GPT (decoder-only) |
|---|---|---|
| Blocks | Encoder stack + decoder stack | Single stack |
| Attention types | 3 (encoder self, decoder self, cross) | **1** (masked self-attention only) |
| Attention direction | Encoder bidirectional, decoder causal | **Causal everywhere** |
| Cross-attention | Yes — bridges encoder to decoder | **None** |
| Trained for | Supervised seq2seq (paired data) | Self-supervised next-token prediction (raw text) |
| Norm placement | Post-Norm | **Pre-Norm** |
| Activation | ReLU | **GELU** |
| Positional encoding | Fixed sinusoidal | **Learned** positional embeddings |

### The single most important architectural point

**The only thing that makes an attention block "causal" is a lower-triangular mask.** The weight matrices, the FFN, the LayerNorms, the residual connections — all identical. Verified in `transformer_block.py`, where the same block class produced encoder-style or decoder-style behaviour depending purely on whether a mask was passed:

```
Head 0 attention weights with causal mask:
[[1.     0.     0.     0.     0.     0.    ]
 [0.5047 0.4953 0.     0.     0.     0.    ]
 [0.3281 0.3292 0.3427 0.     0.     0.    ]
 [0.2493 0.2436 0.2528 0.2544 0.     0.    ]
 [0.1981 0.2003 0.2022 0.1968 0.2026 0.    ]
 [0.1608 0.1635 0.1703 0.1621 0.1724 0.1709]]

Upper triangle (future positions) all exactly zero: True
```

Position 0 attends only to itself. Position 5 attends to all six positions. Every entry above the diagonal is exactly 0.0 — masked positions were set to −∞ before softmax, and exp(−∞) = 0.

### Why the mask is what makes training work

Without causal masking, next-token prediction would be trivially degenerate: the model could simply attend to the token it's supposed to predict and copy it, achieving zero loss while learning nothing. The mask makes the task genuinely predictive.

It also enables the training parallelism described in Part 3: because each position can only see earlier positions, one forward pass over a 1,000-token sequence produces 1,000 independent, valid training signals simultaneously — "given tokens 1..k, predict k+1" for every k at once. This is why decoder-only models scale so efficiently on raw text.

### Why decoder-only won for general-purpose LLMs

1. **Training data is unconstrained.** Encoder-decoder needs paired input/output data; decoder-only needs only raw text, which exists in effectively unlimited quantity.
2. **The architecture is simpler.** One stack, one attention type — easier to scale and optimize.
3. **Task-generality.** Any task expressible as text-in → text-out becomes next-token prediction, so a single model handles translation, summarization, Q&A, and code without task-specific heads.
4. **BERT keeps the other half.** BERT is the mirror image: encoder-only, bidirectional, trained with masked-token prediction. Bidirectional context makes it better for *understanding* tasks (classification, NER, embeddings — as used in Day 3), but it cannot generate autoregressively. GPT's causal masking is exactly what enables generation.

---

## Deliverables

| File | Contents |
|---|---|
| `scaled_dot_product_attention.py` | Attention + multi-head + causal masking, pure NumPy, manually verified |
| `positional_encoding.py` | Sinusoidal PE with all four properties verified numerically |
| `transformer_block.py` | Complete block: attention, FFN, LayerNorm, residuals, Pre/Post-Norm, parameter counts |
| `outputs/*.txt` | Full captured output from every script run |

---

## Summary: How Day 4 Closes the Arc

Days 1–3 each exposed a specific limitation through measurement. Day 4 shows how one architecture resolves all of them:

| Limitation measured earlier | How the Transformer resolves it |
|---|---|
| BoW/TF-IDF: no meaning, no word order (Days 1–2) | Learned embeddings + positional encoding + attention |
| Word2Vec: one frozen vector per word (Day 2) | Layers recompute each token's representation per context |
| N-grams: fixed context window (Day 3) | Attention spans the full sequence, O(1) path length |
| LSTM: gradient vanished to 0.000000 (Day 3) | Residual connections give the gradient an identity highway |
| LSTM: strictly sequential, no parallelism (Day 3) | All positions computed in one parallel matrix operation |

The remaining cost of this design is the O(n²) attention complexity — which is exactly why "context window" is the defining engineering constraint of the current LLM era, and why efficient-attention research (FlashAttention, sparse attention, linear attention) is such an active field.
