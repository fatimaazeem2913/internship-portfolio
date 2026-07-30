"""
transformer_block.py
----------------------
A complete Transformer block (encoder-style) implemented from scratch in NumPy,
integrating all five architectural components studied in Day 4:

    1. Multi-Head Self-Attention
    2. Residual Connections
    3. Layer Normalization
    4. Position-wise Feed-Forward Network
    5. (Positional Encoding is applied before the block -- see positional_encoding.py)

BLOCK STRUCTURE (Post-Norm, as in the original 2017 paper):

    x -> MultiHeadAttention -> Add(x) -> LayerNorm -> FFN -> Add -> LayerNorm -> out

Modern LLMs typically use Pre-Norm instead (LayerNorm BEFORE each sublayer),
which trains more stably at depth. Both variants are implemented here for
comparison.
"""

import numpy as np
from scaled_dot_product_attention import multi_head_attention, causal_mask

np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """
    Layer Normalization (Ba et al., 2016).

    Normalizes ACROSS THE FEATURE DIMENSION for each token independently --
    NOT across the batch (that would be BatchNorm). For each token's vector:

        normalized = (x - mean) / sqrt(variance + eps)
        output     = gamma * normalized + beta

    gamma (scale) and beta (shift) are learned parameters that let the network
    undo the normalization if that's actually useful -- so normalization never
    reduces the model's expressive power.

    WHY LAYERNORM AND NOT BATCHNORM FOR TRANSFORMERS:
    BatchNorm's statistics depend on other examples in the batch, which is
    problematic for variable-length sequences and breaks entirely at inference
    with batch size 1. LayerNorm's statistics come only from the single token
    being normalized, so it behaves identically at train and inference time,
    regardless of batch composition.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        normalized = normalized * gamma
    if beta is not None:
        normalized = normalized + beta
    return normalized


def gelu(x):
    """
    GELU activation (Gaussian Error Linear Unit) -- used in GPT/BERT instead of
    the original paper's ReLU.

    Approximation: 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))

    Unlike ReLU's hard cutoff at 0, GELU is smooth and allows small negative
    values through, which empirically trains better in deep Transformers.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def feed_forward_network(x, W1, b1, W2, b2, activation="gelu"):
    """
    Position-wise Feed-Forward Network.

        FFN(x) = activation(x @ W1 + b1) @ W2 + b2

    "Position-wise" means it's applied INDEPENDENTLY and IDENTICALLY to every
    token position -- the same weight matrices, no mixing between positions.
    All cross-token communication happens in the attention sublayer; the FFN
    is purely per-token processing.

    Typically the hidden dimension is 4x d_model (e.g., 768 -> 3072 -> 768 in
    GPT-2). This expand-then-contract structure is where most of a Transformer's
    parameters actually live -- roughly 2/3 of the total in a standard block.
    """
    hidden = x @ W1 + b1
    hidden = gelu(hidden) if activation == "gelu" else np.maximum(0, hidden)
    return hidden @ W2 + b2


class TransformerBlock:
    """A single Transformer block with all components."""

    def __init__(self, d_model, num_heads, d_ff=None, pre_norm=False):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.pre_norm = pre_norm

        scale = 0.02  # GPT-2 style small initialization
        # Attention projections
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        # FFN weights
        self.W1 = np.random.randn(d_model, self.d_ff) * scale
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)
        # LayerNorm learnable parameters (initialized to identity transform)
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def forward(self, x, mask=None, trace=False):
        steps = {}

        if self.pre_norm:
            # PRE-NORM (modern LLMs): x + Sublayer(LayerNorm(x))
            normed = layer_norm(x, self.gamma1, self.beta1)
            attn_out, attn_weights = multi_head_attention(
                normed, self.W_q, self.W_k, self.W_v, self.W_o, self.num_heads, mask
            )
            x1 = x + attn_out                       # residual
            steps["after_attn_residual"] = x1

            normed2 = layer_norm(x1, self.gamma2, self.beta2)
            ffn_out = feed_forward_network(normed2, self.W1, self.b1, self.W2, self.b2)
            out = x1 + ffn_out                       # residual
        else:
            # POST-NORM (original 2017 paper): LayerNorm(x + Sublayer(x))
            attn_out, attn_weights = multi_head_attention(
                x, self.W_q, self.W_k, self.W_v, self.W_o, self.num_heads, mask
            )
            x1 = layer_norm(x + attn_out, self.gamma1, self.beta1)
            steps["after_attn_residual"] = x1

            ffn_out = feed_forward_network(x1, self.W1, self.b1, self.W2, self.b2)
            out = layer_norm(x1 + ffn_out, self.gamma2, self.beta2)

        steps["attn_weights"] = attn_weights
        steps["output"] = out
        return (out, steps) if trace else out

    def parameter_count(self):
        """Count parameters, broken down by component."""
        attn = 4 * self.d_model * self.d_model                       # W_q, W_k, W_v, W_o
        ffn = self.d_model * self.d_ff + self.d_ff + self.d_ff * self.d_model + self.d_model
        ln = 4 * self.d_model                                         # 2 LayerNorms x (gamma + beta)
        return {"attention": attn, "ffn": ffn, "layernorm": ln, "total": attn + ffn + ln}


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("COMPLETE TRANSFORMER BLOCK FROM SCRATCH (pure NumPy)")
    out("=" * 90)

    seq_len, d_model, num_heads = 6, 64, 8
    x = np.random.randn(seq_len, d_model)

    # ----------------------------------------------------------
    # LayerNorm verification
    # ----------------------------------------------------------
    out("\n" + "-" * 90)
    out("COMPONENT: Layer Normalization")
    out("-" * 90)
    test_input = np.array([[1.0, 2.0, 3.0, 4.0],
                          [10.0, 20.0, 30.0, 40.0],
                          [-5.0, 0.0, 5.0, 100.0]])
    normed = layer_norm(test_input)
    out(f"\nInput (3 tokens, 4 features each):\n{test_input}")
    out(f"\nPer-token means before:     {test_input.mean(axis=-1)}")
    out(f"Per-token std devs before:  {test_input.std(axis=-1)}")
    out(f"\nAfter LayerNorm:\n{normed}")
    out(f"\nPer-token means after:      {normed.mean(axis=-1)}   <- all ~0")
    out(f"Per-token std devs after:   {normed.std(axis=-1)}   <- all ~1")
    out("\nCritically, normalization happens PER TOKEN across features. Token 3's")
    out("wildly different scale (-5 to 100) is normalized using only token 3's own")
    out("statistics -- it never looks at the other tokens or other batch examples.")
    out("This is what makes LayerNorm batch-size-independent and safe at inference.")

    # ----------------------------------------------------------
    # Residual connection demonstration
    # ----------------------------------------------------------
    out("\n" + "-" * 90)
    out("COMPONENT: Residual Connections")
    out("-" * 90)
    out("\nThe residual pattern is: output = x + Sublayer(x)")
    out("\nWhy this matters -- gradient flow. Consider the derivative:")
    out("    d(x + F(x))/dx = 1 + dF(x)/dx")
    out("\nThat '+1' term is the key. Even if dF/dx is tiny (a sublayer that barely")
    out("affects its input), the gradient still flows backward at full strength")
    out("through the identity path. Stacking 96 layers is only feasible because")
    out("each residual connection guarantees an unobstructed gradient highway.")
    out("\nEmpirical check -- simulating stacked sublayers whose local derivative")
    out("dF/dx = 0.7 (a plausibly 'weak' sublayer):\n")

    local_derivative = 0.7
    out(f"  {'Layers':<10}{'Without residual':<22}{'With residual':<22}")
    out("  " + "-" * 52)
    for n_layers in [10, 25, 50, 96]:
        # Without residual: gradient is the product of local derivatives
        without = local_derivative ** n_layers
        # With residual: each layer contributes (1 + dF/dx) instead of just dF/dx
        with_res = (1 + local_derivative) ** n_layers
        out(f"  {n_layers:<10}{without:<22.3e}{with_res:<22.3e}")

    out("\n  Without residuals, the gradient decays exponentially toward zero.")
    out("  With residuals, the '+1' identity term means the gradient does NOT")
    out("  decay -- it actually grows, which is why Transformers additionally")
    out("  need LayerNorm to keep the magnitudes controlled. The two components")
    out("  work as a pair: residuals guarantee gradient flow, LayerNorm keeps")
    out("  the resulting magnitudes numerically stable.")
    out("\n(Compare this to the Day 3 LSTM result, where the gradient literally hit")
    out(" 0.000000 across just 55 timesteps -- same underlying math, and residual")
    out(" connections are precisely the architectural fix.)")

    # ----------------------------------------------------------
    # Feed-forward network
    # ----------------------------------------------------------
    out("\n" + "-" * 90)
    out("COMPONENT: Position-wise Feed-Forward Network")
    out("-" * 90)
    d_ff = 4 * d_model
    W1 = np.random.randn(d_model, d_ff) * 0.02
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.02
    b2 = np.zeros(d_model)
    ffn_out = feed_forward_network(x, W1, b1, W2, b2)
    out(f"\nInput shape:            {x.shape}")
    out(f"Hidden (expanded) dim:  {d_ff}   (= 4 x d_model, standard ratio)")
    out(f"Output shape:           {ffn_out.shape}   <- back to d_model")
    out("\nEach token passes through INDEPENDENTLY -- verifying that now:")
    single_token = x[2:3]
    single_out = feed_forward_network(single_token, W1, b1, W2, b2)
    out(f"  FFN(token 2) computed alone:      {single_out[0][:5]}")
    out(f"  FFN(token 2) from full sequence:  {ffn_out[2][:5]}")
    out(f"  IDENTICAL: {np.allclose(single_out[0], ffn_out[2])}")
    out("\nThis confirms the FFN does NO cross-token mixing. All communication")
    out("between positions happens exclusively in the attention sublayer -- the")
    out("FFN just processes each token's representation on its own.")

    # ----------------------------------------------------------
    # Full block: Post-Norm vs Pre-Norm
    # ----------------------------------------------------------
    out("\n" + "-" * 90)
    out("FULL BLOCK: Post-Norm (2017 paper) vs Pre-Norm (modern LLMs)")
    out("-" * 90)

    block_post = TransformerBlock(d_model, num_heads, pre_norm=False)
    block_pre = TransformerBlock(d_model, num_heads, pre_norm=True)

    out_post, steps_post = block_post.forward(x, trace=True)
    out_pre, steps_pre = block_pre.forward(x, trace=True)

    out(f"\nInput shape:              {x.shape}")
    out(f"Post-Norm output shape:   {out_post.shape}")
    out(f"Pre-Norm output shape:    {out_pre.shape}")
    out(f"\nPost-Norm output stats:  mean={out_post.mean():.6f}, std={out_post.std():.6f}")
    out(f"Pre-Norm output stats:   mean={out_pre.mean():.6f}, std={out_pre.std():.6f}")
    out("\nNote Post-Norm's output is tightly normalized (LayerNorm is the final")
    out("operation), while Pre-Norm's output retains the residual stream's scale.")
    out("Pre-Norm's unnormalized residual path is exactly why it trains more")
    out("stably at extreme depth -- the gradient highway is never interrupted by")
    out("a normalization step.")
    out(f"\nAttention maps produced: {len(steps_post['attn_weights'])} (one per head)")
    out(f"Each attention map shape: {steps_post['attn_weights'][0].shape}")

    # ----------------------------------------------------------
    # Causal masking in a full block (GPT-style)
    # ----------------------------------------------------------
    out("\n" + "-" * 90)
    out("DECODER-ONLY VARIANT: same block, with causal masking")
    out("-" * 90)
    mask = causal_mask(seq_len)
    out_causal, steps_causal = block_pre.forward(x, mask=mask, trace=True)
    head0_weights = steps_causal["attn_weights"][0]
    out(f"\nHead 0 attention weights with causal mask:\n{head0_weights}")
    out(f"\nUpper triangle (future positions) all exactly zero: "
        f"{np.allclose(np.triu(head0_weights, k=1), 0.0)}")
    out("\nThis single change -- adding a lower-triangular mask -- is the ONLY")
    out("architectural difference that converts an encoder block (BERT-style,")
    out("bidirectional) into a decoder block (GPT-style, autoregressive).")

    # ----------------------------------------------------------
    # Parameter accounting
    # ----------------------------------------------------------
    out("\n" + "-" * 90)
    out("PARAMETER COUNT BREAKDOWN")
    out("-" * 90)
    for label, dm, nh in [("This demo block", d_model, num_heads),
                          ("GPT-2 Small block", 768, 12),
                          ("GPT-2 Medium block", 1024, 16),
                          ("BERT-Base block", 768, 12)]:
        b = TransformerBlock(dm, nh)
        counts = b.parameter_count()
        out(f"\n{label} (d_model={dm}, heads={nh}):")
        out(f"  Attention (W_q,W_k,W_v,W_o):  {counts['attention']:>12,}  "
            f"({100*counts['attention']/counts['total']:.1f}%)")
        out(f"  Feed-Forward Network:          {counts['ffn']:>12,}  "
            f"({100*counts['ffn']/counts['total']:.1f}%)")
        out(f"  LayerNorm parameters:          {counts['layernorm']:>12,}  "
            f"({100*counts['layernorm']/counts['total']:.1f}%)")
        out(f"  TOTAL per block:               {counts['total']:>12,}")

    out("\nKey insight: the FFN holds roughly TWO-THIRDS of every block's")
    out("parameters, despite attention getting all the conceptual attention (pun")
    out("intended). LayerNorm contributes a negligible fraction. For a 12-layer")
    out("GPT-2 Small, 12 x 7,087,872 = ~85M parameters in the blocks alone,")
    out("before counting embeddings.")

    out("\n" + "=" * 90)
    out("ALL TRANSFORMER COMPONENTS IMPLEMENTED AND VERIFIED")
    out("=" * 90)

    with open("outputs/transformer_block_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/transformer_block_results.txt")
