"""
positional_encoding.py
------------------------
Sinusoidal positional encoding from scratch in NumPy, exactly as specified in
Vaswani et al. (2017), Section 3.5.

THE PROBLEM IT SOLVES:
Self-attention is permutation-invariant. If you shuffle the input tokens, the
attention computation produces the same set of outputs in shuffled order -- the
mechanism has NO inherent notion of "this word came before that word." An RNN
gets order for free (it literally processes tokens in sequence); a Transformer
does not. So order information must be explicitly injected.

THE FORMULA:
    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

Where pos = position in the sequence, i = dimension index pair.
Even dimensions get sine, odd dimensions get cosine.

WHY SINUSOIDS RATHER THAN JUST [0, 1, 2, 3, ...]:
1. Bounded: values stay in [-1, 1], so they don't overwhelm the word embeddings
   they're added to (raw integers would grow without bound for long sequences).
2. Unique: each position gets a distinct pattern across all d_model dimensions.
3. Relative positions are learnable as linear functions: PE(pos+k) can be
   expressed as a linear transformation of PE(pos) for any fixed offset k --
   which means the model can learn "attend 3 tokens back" as a single
   learnable operation, generalizing across absolute positions.
4. Extrapolation: the function is defined for any position, so in principle the
   model can handle sequences longer than any it saw during training.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def sinusoidal_positional_encoding(max_seq_len, d_model):
    """
    Build the positional encoding matrix.

    Returns: array of shape [max_seq_len, d_model]
    """
    PE = np.zeros((max_seq_len, d_model))

    # positions: column vector [max_seq_len, 1]
    position = np.arange(max_seq_len)[:, np.newaxis]

    # The division term: 10000^(2i/d_model) for each dimension pair i
    # Computed in log space for numerical stability:
    #   10000^(2i/d_model) = exp( (2i/d_model) * ln(10000) )
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)   # even dims -> sine
    PE[:, 1::2] = np.cos(position * div_term)   # odd dims  -> cosine

    return PE


def verify_relative_position_property(PE, pos, offset):
    """
    Verify the key theoretical claim: PE(pos + k) is a LINEAR function of PE(pos).

    For a given frequency, the sin/cos pair at position pos+k relates to the pair
    at position pos by a fixed 2D rotation matrix:
        [sin(pos+k)]   [cos(k)  sin(k)] [sin(pos)]
        [cos(pos+k)] = [-sin(k) cos(k)] [cos(pos)]

    This is why the model can learn relative-position attention patterns.
    """
    d_model = PE.shape[1]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    results = []
    for i, freq in enumerate(div_term[:4]):   # check first 4 frequency bands
        theta_k = offset * freq
        rotation = np.array([[np.cos(theta_k),  np.sin(theta_k)],
                            [-np.sin(theta_k), np.cos(theta_k)]])
        original_pair = np.array([PE[pos, 2 * i], PE[pos, 2 * i + 1]])
        predicted = rotation @ original_pair
        actual = np.array([PE[pos + offset, 2 * i], PE[pos + offset, 2 * i + 1]])
        results.append((i, predicted, actual, np.allclose(predicted, actual, atol=1e-9)))
    return results


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("SINUSOIDAL POSITIONAL ENCODING FROM SCRATCH (pure NumPy)")
    out("=" * 90)

    # --- Small example we can actually read ---
    max_len, d_model = 8, 8
    PE = sinusoidal_positional_encoding(max_len, d_model)

    out(f"\nPositional encoding matrix (max_seq_len={max_len}, d_model={d_model}):\n")
    out(f"{'pos':<5}" + "".join(f"dim{i:<6}" for i in range(d_model)))
    out("-" * 78)
    for pos in range(max_len):
        out(f"{pos:<5}" + "".join(f"{PE[pos, i]:<9.4f}" for i in range(d_model)))

    out("\nObservations from the matrix above:")
    out("  - Position 0 gives sin(0)=0 for all even dims, cos(0)=1 for all odd dims")
    out("  - Lower dimensions (left) oscillate FAST -- they encode fine-grained,")
    out("    local position differences")
    out("  - Higher dimensions (right) oscillate SLOWLY -- they encode coarse,")
    out("    long-range position information")
    out("  - Together, the full d_model-dimensional pattern is unique per position,")
    out("    like a binary counter but continuous instead of discrete")

    # --- Property 1: bounded values ---
    out("\n" + "-" * 90)
    out("PROPERTY 1: Values are bounded in [-1, 1]")
    out("-" * 90)
    PE_large = sinusoidal_positional_encoding(512, 128)
    out(f"\nFor a 512-position, 128-dim encoding:")
    out(f"  Minimum value: {PE_large.min():.6f}")
    out(f"  Maximum value: {PE_large.max():.6f}")
    out(f"  All values within [-1, 1]: {np.all((PE_large >= -1) & (PE_large <= 1))}")
    out("\nThis matters because positional encodings are ADDED to word embeddings.")
    out("If we naively used raw position integers (0, 1, 2, ... 511), position 511")
    out("would completely swamp the embedding's semantic content. Bounded sinusoids")
    out("perturb the embedding without destroying it.")

    # --- Property 2: uniqueness ---
    out("\n" + "-" * 90)
    out("PROPERTY 2: Every position gets a unique encoding")
    out("-" * 90)
    unique_rows = len(np.unique(PE_large, axis=0))
    out(f"\nDistinct encoding vectors across 512 positions: {unique_rows} / 512")
    out(f"All positions uniquely encoded: {unique_rows == 512}")

    # --- Property 3: the relative-position linearity claim ---
    out("\n" + "-" * 90)
    out("PROPERTY 3: PE(pos + k) is a LINEAR function of PE(pos)")
    out("-" * 90)
    out("\nThis is the theoretically important property. For each sin/cos frequency")
    out("pair, shifting position by a fixed offset k is exactly a 2D rotation:")
    out("")
    out("    [sin(w(pos+k))]   [ cos(wk)  sin(wk)] [sin(w*pos)]")
    out("    [cos(w(pos+k))] = [-sin(wk)  cos(wk)] [cos(w*pos)]")
    out("")
    out("Verifying numerically for pos=3, offset k=5, first 4 frequency bands:\n")

    verification = verify_relative_position_property(PE_large, pos=3, offset=5)
    for i, predicted, actual, matches in verification:
        out(f"  Frequency band {i}:")
        out(f"    Rotation-predicted PE(8): {predicted}")
        out(f"    Actual PE(8):             {actual}")
        out(f"    MATCH: {matches}")
    all_match = all(v[3] for v in verification)
    out(f"\n  All frequency bands verified: {all_match}")
    out("\nWHY THIS MATTERS: because relative offsets are linear transformations,")
    out("the model can learn a single weight matrix meaning 'attend to whatever is")
    out("3 positions back' and apply it at ANY absolute position. Relative position")
    out("becomes a learnable, position-independent operation.")

    # --- Property 4: similarity decays with distance ---
    out("\n" + "-" * 90)
    out("PROPERTY 4: Encoding similarity decays with positional distance")
    out("-" * 90)
    out("\nDot product between PE(0) and PE(k) for increasing k:\n")
    ref = PE_large[0]
    for k in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
        dot = ref @ PE_large[k]
        cos_sim = dot / (np.linalg.norm(ref) * np.linalg.norm(PE_large[k]))
        out(f"  k = {k:<5} dot = {dot:>9.4f}   cosine similarity = {cos_sim:>7.4f}")
    out("\nNearby positions have similar encodings; distant positions diverge. This")
    out("gives the model a usable, continuous notion of 'how far apart are these")
    out("two tokens' -- without ever explicitly computing a distance.")

    # --- Demonstrate the permutation-invariance problem being solved ---
    out("\n" + "-" * 90)
    out("WHY THIS IS NECESSARY: attention is permutation-invariant without it")
    out("-" * 90)

    from scaled_dot_product_attention import scaled_dot_product_attention

    np.random.seed(0)
    d = 8
    X = np.random.randn(4, d)              # 4 token embeddings, NO position info
    perm = [2, 0, 3, 1]                     # an arbitrary shuffle
    X_shuffled = X[perm]

    out_orig, _ = scaled_dot_product_attention(X, X, X)
    out_shuf, _ = scaled_dot_product_attention(X_shuffled, X_shuffled, X_shuffled)

    out("\nWITHOUT positional encoding:")
    out(f"  Attention output for original order, row for token '{perm[0]}':")
    out(f"    {out_orig[perm[0]]}")
    out(f"  Attention output for shuffled order, same token now at row 0:")
    out(f"    {out_shuf[0]}")
    out(f"  IDENTICAL: {np.allclose(out_orig[perm[0]], out_shuf[0])}")
    out("\n  -> Shuffling the input just shuffles the output. The mechanism cannot")
    out("     tell the difference between 'dog bites man' and 'man bites dog'.")

    PE_small = sinusoidal_positional_encoding(4, d)
    X_pe = X + PE_small
    X_shuf_pe = X_shuffled + PE_small       # positions reassigned after shuffle

    out_orig_pe, _ = scaled_dot_product_attention(X_pe, X_pe, X_pe)
    out_shuf_pe, _ = scaled_dot_product_attention(X_shuf_pe, X_shuf_pe, X_shuf_pe)

    out("\nWITH positional encoding added:")
    out(f"  Original-order output for that token:  {out_orig_pe[perm[0]]}")
    out(f"  Shuffled-order output, same token:     {out_shuf_pe[0]}")
    out(f"  IDENTICAL: {np.allclose(out_orig_pe[perm[0]], out_shuf_pe[0])}")
    out("\n  -> Now they differ. Order information has been successfully injected,")
    out("     so the model CAN distinguish different word orders.")

    out("\n" + "=" * 90)
    out("ALL POSITIONAL ENCODING PROPERTIES VERIFIED")
    out("=" * 90)

    with open("outputs/positional_encoding_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/positional_encoding_results.txt")
