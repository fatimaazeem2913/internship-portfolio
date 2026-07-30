# Transformer Architecture & Attention Mechanisms – Day 4 Internship

## Project Overview

This project was completed as part of Day 4 internship tasks. The objective was to build a solid technical understanding of how Transformer models process sequences — deep enough to reason about LLM behaviour at the architecture level rather than treating models as black boxes.

The work covers a close reading of "Attention Is All You Need" (Vaswani et al., 2017), a from-scratch NumPy implementation of scaled dot-product attention and every other core Transformer component, a full architectural diagram, precise clarification of commonly-confused terminology, and a study of GPT-2's decoder-only design via Andrej Karpathy's nanoGPT.

Every implementation is pure NumPy — no PyTorch, no TensorFlow — so every operation is explicit and verifiable by hand.

---

## Objectives

- Read and document "Attention Is All You Need": the problem it solved, the core innovation, and why self-attention replaced RNNs.
- Document each Transformer component: Multi-Head Self-Attention, Feed-Forward Network, Layer Normalization, Residual Connections, Positional Encoding.
- Implement scaled dot-product attention from scratch in NumPy and verify against manual computation.
- Create a full architectural diagram: Input → Tokenization → Embeddings → Positional Encoding → N Transformer Layers → Output.
- Clearly distinguish: Tokens vs Words, Training vs Inference, Context Window vs Memory, Embeddings vs Tokens.
- Study GPT-2 via nanoGPT and document how decoder-only differs from the original encoder-decoder design.

---

## Technologies Used

- Python 3
- NumPy (the only dependency — deliberately no deep learning frameworks)

---

## Project Structure

```
day-4-transformer-architecture
|
|-- README.md
|-- REPORT.md
|
|-- scaled_dot_product_attention.py
|-- positional_encoding.py
|-- transformer_block.py
|
|-- outputs
    |-- attention_from_scratch_results.txt
    |-- positional_encoding_results.txt
    `-- transformer_block_results.txt
```

---

## Tasks Performed

### 1. Scaled Dot-Product Attention from Scratch

Implemented `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` in pure NumPy with every intermediate step exposed. Includes numerically stable softmax, causal masking, and full multi-head attention.

Verified by computing one attention row entirely by hand (dot products → scaling → softmax → weighted sum) and confirming it matched the function output exactly.

**Output:** `outputs/attention_from_scratch_results.txt`

### 2. Sinusoidal Positional Encoding from Scratch

Implemented the paper's sinusoidal formula and verified all four of its theoretical properties numerically, including the rotation-matrix relationship that makes relative positions learnable as linear transformations.

Also demonstrated *why* positional encoding is necessary by showing that attention output is bit-for-bit identical under input shuffling without it, and differs with it.

**Output:** `outputs/positional_encoding_results.txt`

### 3. Complete Transformer Block

Implemented a full block integrating all five components, in both Post-Norm (original paper) and Pre-Norm (modern LLM) variants, with causal masking support and a parameter-count breakdown across real model configurations.

**Output:** `outputs/transformer_block_results.txt`

### 4. Architecture Diagram

Full pipeline diagram: Raw text → Tokenization → Token embeddings → Positional encoding → N × (Multi-head attention → Add & Norm → FFN → Add & Norm) → Linear projection → Softmax.

### 5. Terminology Documentation

Precise distinctions documented in `REPORT.md` Part 3 for Tokens vs Words, Training vs Inference, Context Window vs Memory, and Embeddings vs Tokens.

### 6. nanoGPT / Decoder-Only Study

Documented in `REPORT.md` Part 4: the three attention mechanisms in the original design, which two GPT discards, and why a single lower-triangular mask is the entire architectural difference between an encoder and a decoder block.

---

## Results

- **Attention implementation verified by hand.** Manually computed attention weights `[0.4011, 0.1978, 0.4011]` for a toy query matched the function output exactly, as did the resulting output vector.
- **Attention weights confirmed to form a valid probability distribution:** all values in [0,1], every row summing to exactly 1.0.
- **The √d_k scaling justified empirically:** at d_k=512, unscaled softmax produced a max attention weight of 1.0000 (fully saturated, zero gradient); with scaling it stayed at 0.6265.
- **Causal masking verified:** upper triangle of the attention matrix exactly zero, confirming no token can attend to future positions.
- **Multi-head attention produced 8 genuinely different attention distributions** over identical input, confirming heads can specialize.
- **Positional encoding's necessity proven directly:** without it, shuffling the input produced bit-for-bit identical attention output; with it, output differed.
- **All four positional-encoding properties verified numerically,** including the rotation-matrix relative-position property matching to within 1e-9.
- **FFN confirmed to do no cross-token mixing:** running it on an isolated token gave an identical result to running it within the full sequence.
- **Parameter breakdown computed:** the FFN holds 66.7% of every block's parameters, attention 33.3%, LayerNorm ~0.0%.

---

## Observations

- The √d_k scaling is not a minor implementation detail — it is the difference between a trainable and an untrainable attention mechanism. The empirical demonstration (max weight 1.0000 unscaled vs 0.6265 scaled at d_k=512) makes this concrete.
- Self-attention's complexity is O(n²·d) per layer, which is *worse* than an RNN's O(n·d²) for long sequences. The paper's bet was that quadratic-but-parallel beats linear-but-sequential on GPU hardware — and that bet is also precisely why context window length is the central engineering constraint of the LLM era.
- Residual connections and LayerNorm must be understood as a pair, not independently. Residuals alone cause gradient magnitudes to *grow* exponentially with depth (measured: 1.328e+22 across 96 layers); LayerNorm is what keeps them numerically stable.
- The FFN holds two-thirds of every block's parameters despite receiving a small fraction of the conceptual discussion. Attention is where the interesting *computation* happens; the FFN is where most of the *capacity* lives.
- Attention is permutation-invariant without positional encoding — meaning a raw Transformer genuinely cannot distinguish "dog bites man" from "man bites dog." This was verified directly rather than taken on faith.
- A single lower-triangular mask is the *only* architectural difference between an encoder block and a decoder block. The weights, FFN, LayerNorms, and residuals are identical.
- A Transformer has no memory — it has a context window. Everything that feels like memory in a chat interface is the application re-sending prior turns as input on every request.

---

## Challenges Encountered

- Getting the manual hand-verification of attention to match the vectorized implementation required care with the order of operations — specifically, that scaling by √d_k happens *before* softmax, not after. Verifying one full row by hand was the check that confirmed the implementation was correct rather than merely plausible-looking.
- The initial residual-connection gradient demonstration was written in a way that didn't cleanly isolate the effect being measured. It was rewritten to compare `(dF/dx)^n` against `(1 + dF/dx)^n` directly across increasing depth, which shows both the vanishing *and* the growth behaviour honestly — and motivates why LayerNorm is required alongside residuals rather than presenting residuals as a complete solution.
- Demonstrating permutation-invariance required carefully constructing the test so that the *same* token was tracked before and after shuffling, rather than just comparing output matrices (which would trivially differ in row order regardless).

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-4-transformer-architecture
```

Only NumPy is required:
```
pip install numpy
```

Run the scripts in order (`transformer_block.py` imports from `scaled_dot_product_attention.py`):
```
python3 scaled_dot_product_attention.py
python3 positional_encoding.py
python3 transformer_block.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- Why self-attention replaced recurrence: not lower FLOPs, but O(1) sequential operations and O(1) path length between any two positions, which together enable both parallel training and long-range dependency learning.
- The precise role of every component in a Transformer block, and the division of labour between attention (cross-token communication) and the FFN (per-token processing).
- Why the √d_k scaling exists, demonstrated by measuring softmax saturation at increasing d_k.
- Why residual connections and LayerNorm are architecturally inseparable.
- Why positional encoding is mandatory rather than optional, proven by demonstrating permutation-invariance.
- The mathematical reason relative positions are learnable in sinusoidal encoding — the 2D rotation-matrix relationship between PE(pos) and PE(pos+k).
- That a single causal mask is the entire architectural difference between BERT-style encoders and GPT-style decoders.
- Precise, defensible distinctions between tokens and words, training and inference, context window and memory, and embeddings and tokens — including why a Transformer has no memory at all.
- How the O(n²) cost of attention directly explains why context window length is the defining engineering constraint of modern LLMs.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 4
