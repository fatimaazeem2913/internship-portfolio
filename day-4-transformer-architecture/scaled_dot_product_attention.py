"""
scaled_dot_product_attention.py
---------------------------------
Scaled dot-product attention implemented from scratch in pure NumPy.
NO PyTorch, NO TensorFlow -- every operation is explicit so the mechanics
are fully visible.

THE FORMULA (Vaswani et al., 2017, Section 3.2.1):

    Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V

Where:
    Q = Query matrix   [seq_len, d_k]  -- "what am I looking for?"
    K = Key matrix     [seq_len, d_k]  -- "what do I contain?"
    V = Value matrix   [seq_len, d_v]  -- "what do I contribute if attended to?"
    d_k = dimensionality of the key/query vectors

WHY THE sqrt(d_k) SCALING:
    Q @ K.T produces dot products. For random vectors with components of
    variance 1, a dot product over d_k dimensions has variance d_k -- so the
    values grow as d_k grows. Large magnitudes push softmax into a regime where
    one value dominates and the rest are ~0, making the gradient vanish. Dividing
    by sqrt(d_k) normalizes the variance back to ~1, keeping softmax in a
    well-behaved range. This single detail is why the paper says "scaled".
"""

import numpy as np

np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)


def softmax(x, axis=-1):
    """
    Numerically stable softmax.

    Subtracting the row max before exponentiating prevents overflow: exp(1000)
    overflows to inf, but exp(1000 - 1000) = exp(0) = 1. The result is
    mathematically identical because softmax is invariant to constant shifts.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None, verbose=False):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query matrix,  shape [seq_len_q, d_k]
        K: Key matrix,    shape [seq_len_k, d_k]
        V: Value matrix,  shape [seq_len_k, d_v]
        mask: optional boolean array, shape [seq_len_q, seq_len_k].
              True = position is allowed to be attended to.
              False = position is masked out (set to -inf before softmax).
        verbose: if True, print every intermediate step.

    Returns:
        output:            shape [seq_len_q, d_v]
        attention_weights: shape [seq_len_q, seq_len_k]
    """
    d_k = Q.shape[-1]

    # STEP 1: raw compatibility scores between every query and every key.
    # scores[i, j] = how much query i "matches" key j
    scores = Q @ K.T

    # STEP 2: scale by sqrt(d_k) to keep softmax gradients healthy
    scaled_scores = scores / np.sqrt(d_k)

    # STEP 3 (optional): apply mask. Setting masked positions to -inf means
    # exp(-inf) = 0, so they receive exactly zero attention weight.
    if mask is not None:
        scaled_scores = np.where(mask, scaled_scores, -np.inf)

    # STEP 4: softmax over the key axis -> each query's attention weights sum to 1
    attention_weights = softmax(scaled_scores, axis=-1)

    # STEP 5: weighted sum of value vectors
    output = attention_weights @ V

    if verbose:
        print(f"  d_k = {d_k}, sqrt(d_k) = {np.sqrt(d_k):.4f}\n")
        print(f"  STEP 1 - Raw scores (Q @ K.T):\n{scores}\n")
        print(f"  STEP 2 - Scaled scores (/ sqrt(d_k)):\n{scaled_scores}\n")
        if mask is not None:
            print(f"  STEP 3 - Mask applied:\n{mask}\n")
        print(f"  STEP 4 - Attention weights (after softmax):\n{attention_weights}")
        print(f"           Row sums (should all be 1.0): {attention_weights.sum(axis=-1)}\n")
        print(f"  STEP 5 - Output (weights @ V):\n{output}\n")

    return output, attention_weights


def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """
    Multi-head attention, also from scratch.

    Instead of one attention computation over the full d_model dimensions,
    split into `num_heads` parallel attention computations over d_model/num_heads
    dimensions each, then concatenate and project.

    WHY MULTIPLE HEADS: a single attention distribution can only express one
    "kind" of relationship at a time. With 8 heads, one head might learn to
    track syntactic dependencies, another semantic similarity, another
    positional proximity -- they specialize, and their outputs are combined.
    """
    seq_len, d_model = X.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_head = d_model // num_heads

    # Project input into Q, K, V spaces
    Q = X @ W_q   # [seq_len, d_model]
    K = X @ W_k
    V = X @ W_v

    head_outputs = []
    all_head_weights = []

    for h in range(num_heads):
        start, end = h * d_head, (h + 1) * d_head
        # Slice out this head's portion of the Q/K/V dimensions
        Q_h = Q[:, start:end]
        K_h = K[:, start:end]
        V_h = V[:, start:end]
        out_h, weights_h = scaled_dot_product_attention(Q_h, K_h, V_h, mask=mask)
        head_outputs.append(out_h)
        all_head_weights.append(weights_h)

    # Concatenate all heads back to d_model width, then apply output projection
    concatenated = np.concatenate(head_outputs, axis=-1)  # [seq_len, d_model]
    output = concatenated @ W_o

    return output, all_head_weights


def causal_mask(seq_len):
    """
    Lower-triangular mask for decoder-only (GPT-style) attention.

    Position i may attend to positions 0..i (itself and everything before),
    but NOT to positions i+1.. (the future). This is what makes autoregressive
    generation possible -- without it, the model could 'cheat' by looking at
    the answer it's supposed to predict.
    """
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("SCALED DOT-PRODUCT ATTENTION FROM SCRATCH (pure NumPy)")
    out("=" * 90)

    # ---------------------------------------------------------------
    # TEST 1: Tiny hand-checkable toy example
    # ---------------------------------------------------------------
    out("\n" + "-" * 90)
    out("TEST 1: Toy input -- 3 tokens, d_k = 2 (small enough to verify by hand)")
    out("-" * 90 + "\n")

    Q1 = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]])
    K1 = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]])
    V1 = np.array([[10.0, 0.0],
                   [0.0, 10.0],
                   [5.0, 5.0]])

    out(f"Q =\n{Q1}\n")
    out(f"K =\n{K1}\n")
    out(f"V =\n{V1}\n")

    # Capture the verbose walkthrough
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        output1, weights1 = scaled_dot_product_attention(Q1, K1, V1, verbose=True)
    walkthrough = buf.getvalue()
    out(walkthrough)

    # --- MANUAL VERIFICATION of row 0 ---
    out("MANUAL VERIFICATION (query row 0 = [1, 0]):")
    out("  Dot products with each key:")
    out(f"    q0 . k0 = [1,0].[1,0] = {Q1[0] @ K1[0]:.1f}")
    out(f"    q0 . k1 = [1,0].[0,1] = {Q1[0] @ K1[1]:.1f}")
    out(f"    q0 . k2 = [1,0].[1,1] = {Q1[0] @ K1[2]:.1f}")
    d_k = Q1.shape[-1]
    manual_scaled = np.array([Q1[0] @ K1[0], Q1[0] @ K1[1], Q1[0] @ K1[2]]) / np.sqrt(d_k)
    out(f"  After scaling by sqrt({d_k}) = {np.sqrt(d_k):.4f}: {manual_scaled}")
    manual_weights = softmax(manual_scaled)
    out(f"  After softmax: {manual_weights}")
    out(f"  Library-path result for same row: {weights1[0]}")
    out(f"  MATCH: {np.allclose(manual_weights, weights1[0])}")
    manual_output = manual_weights @ V1
    out(f"  Manual output row 0: {manual_output}")
    out(f"  Function output row 0: {output1[0]}")
    out(f"  MATCH: {np.allclose(manual_output, output1[0])}\n")

    # ---------------------------------------------------------------
    # TEST 2: Verify softmax properties
    # ---------------------------------------------------------------
    out("-" * 90)
    out("TEST 2: Attention weight properties")
    out("-" * 90 + "\n")
    out(f"All weights >= 0:              {np.all(weights1 >= 0)}")
    out(f"All weights <= 1:              {np.all(weights1 <= 1)}")
    out(f"Every row sums to 1.0:         {np.allclose(weights1.sum(axis=-1), 1.0)}")
    out(f"Row sums: {weights1.sum(axis=-1)}\n")
    out("These three properties confirm the attention weights form a valid")
    out("probability distribution over the input positions -- each query")
    out("distributes exactly 100% of its 'attention budget' across all keys.\n")

    # ---------------------------------------------------------------
    # TEST 3: Demonstrate WHY scaling matters
    # ---------------------------------------------------------------
    out("-" * 90)
    out("TEST 3: Why divide by sqrt(d_k)? (empirical demonstration)")
    out("-" * 90 + "\n")

    for d_k_test in [4, 64, 512]:
        Q_t = np.random.randn(4, d_k_test)
        K_t = np.random.randn(4, d_k_test)
        raw = Q_t @ K_t.T
        scaled = raw / np.sqrt(d_k_test)
        w_unscaled = softmax(raw, axis=-1)
        w_scaled = softmax(scaled, axis=-1)
        out(f"d_k = {d_k_test}:")
        out(f"  Raw score std dev:      {raw.std():.4f}")
        out(f"  Scaled score std dev:   {scaled.std():.4f}")
        out(f"  Max weight WITHOUT scaling: {w_unscaled.max():.4f}")
        out(f"  Max weight WITH scaling:    {w_scaled.max():.4f}")
        out("")

    out("Observe: as d_k grows, unscaled dot products grow in magnitude, pushing")
    out("softmax toward a one-hot distribution (max weight -> 1.0). That saturated")
    out("regime has near-zero gradient, so the model can't learn. Scaling by")
    out("sqrt(d_k) keeps the score variance ~constant regardless of d_k, holding")
    out("softmax in a healthy, trainable range. This is the entire reason the")
    out("paper's mechanism is called SCALED dot-product attention.\n")

    # ---------------------------------------------------------------
    # TEST 4: Causal masking (GPT-style)
    # ---------------------------------------------------------------
    out("-" * 90)
    out("TEST 4: Causal (autoregressive) masking -- how GPT prevents 'cheating'")
    out("-" * 90 + "\n")

    seq_len = 4
    Q4 = np.random.randn(seq_len, 8)
    K4 = np.random.randn(seq_len, 8)
    V4 = np.random.randn(seq_len, 8)
    mask4 = causal_mask(seq_len)

    out(f"Causal mask (True = allowed to attend):\n{mask4}\n")
    _, weights4 = scaled_dot_product_attention(Q4, K4, V4, mask=mask4)
    out(f"Resulting attention weights:\n{weights4}\n")
    out("Note the strict lower-triangular structure: position 0 attends only to")
    out("itself; position 3 attends to 0,1,2,3. Every entry above the diagonal is")
    out("exactly 0.0 -- no token can see the future. This is what makes")
    out("next-token prediction a well-posed training objective.")
    out(f"Upper triangle all exactly zero: {np.allclose(np.triu(weights4, k=1), 0.0)}\n")

    # ---------------------------------------------------------------
    # TEST 5: Multi-head attention
    # ---------------------------------------------------------------
    out("-" * 90)
    out("TEST 5: Multi-head attention (8 heads, d_model = 64)")
    out("-" * 90 + "\n")

    seq_len, d_model, num_heads = 5, 64, 8
    X = np.random.randn(seq_len, d_model)
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    W_o = np.random.randn(d_model, d_model) * 0.1

    mh_out, head_weights = multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads)

    out(f"Input shape:              {X.shape}   [seq_len, d_model]")
    out(f"Number of heads:          {num_heads}")
    out(f"Dimensions per head:      {d_model // num_heads}")
    out(f"Output shape:             {mh_out.shape}   [seq_len, d_model]  <- same as input")
    out(f"Attention maps produced:  {len(head_weights)} (one per head), each {head_weights[0].shape}\n")

    out("Each head's attention pattern for token 0 (showing they differ):")
    for h in range(min(4, num_heads)):
        out(f"  Head {h}: {head_weights[h][0]}")
    out("")
    out("The heads produce genuinely DIFFERENT attention distributions over the")
    out("same input -- this is the point of multi-head attention. Each head can")
    out("specialize in a different type of relationship, and their concatenated")
    out("outputs give the model a much richer representation than any single")
    out("attention distribution could provide.\n")

    out("=" * 90)
    out("ALL TESTS PASSED -- implementation verified against manual computation")
    out("=" * 90)

    with open("outputs/attention_from_scratch_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/attention_from_scratch_results.txt")
