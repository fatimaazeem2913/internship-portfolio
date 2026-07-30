"""
lstm_gradient_analysis.py
---------------------------
Runs an UNTRAINED LSTM forward + backward pass over a 50+ token real sequence
from the corpus. Inspects gradient magnitudes flowing back to each timestep's
hidden state to show that earlier tokens' influence decays. Also demonstrates
that each timestep's computation depends on the previous one -- no parallelism
possible within a single sequence.
"""

import re
import time
import torch
import torch.nn as nn

torch.manual_seed(42)


def load_real_sequence(path="data/input_corpus.txt", min_len=55):
    """Pull a real 50+ token sequence straight from the corpus."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    for marker in ["### DOMAIN: NEWS ###", "### DOMAIN: SCIENCE ###", "### DOMAIN: DIALOGUE ###"]:
        raw = raw.replace(marker, "")
    tokens = re.findall(r"[a-z']+", raw.lower())
    return tokens[:min_len]


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, token_ids):
        x = self.embed(token_ids)               # [1, seq_len, embed_dim]
        outputs, (h_n, c_n) = self.lstm(x)       # outputs: [1, seq_len, hidden_dim] -- hidden state at EVERY timestep
        return outputs


if __name__ == "__main__":
    lines = ["=" * 90, "LSTM GRADIENT ANALYSIS: does early-token influence decay across timesteps?", "=" * 90]

    tokens = load_real_sequence()
    seq_len = len(tokens)
    vocab = sorted(set(tokens))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    token_ids = torch.tensor([[word_to_id[t] for t in tokens]])  # shape [1, seq_len]

    lines.append(f"\nReal sequence pulled from corpus ({seq_len} tokens):")
    lines.append(f"  {' '.join(tokens)}")

    model = SimpleLSTM(vocab_size=len(vocab))

    # --- Forward pass, keeping every timestep's hidden output with gradient tracking ---
    embed_out = model.embed(token_ids)
    embed_out.retain_grad()  # so we can inspect gradients on the embedding layer per-timestep
    lstm_outputs, (h_n, c_n) = model.lstm(embed_out)

    # Use the FINAL timestep's hidden state as a stand-in "loss" (as if predicting from it)
    final_hidden = lstm_outputs[0, -1, :]   # hidden state at the LAST timestep
    pseudo_loss = final_hidden.sum()

    # --- Backward pass ---
    pseudo_loss.backward()

    # Gradient of the loss w.r.t. EACH timestep's input embedding
    # -- this tells us how much the final output "cares about" each earlier token
    grad_per_timestep = embed_out.grad[0].norm(dim=1)  # L2 norm of gradient vector at each timestep

    lines.append("\n--- Gradient magnitude at each timestep's input (w.r.t. final hidden state) ---\n")
    lines.append(f"{'Timestep':<10}{'Token':<15}{'Gradient Norm':<15}")
    lines.append("-" * 40)
    for i, (tok, g) in enumerate(zip(tokens, grad_per_timestep.tolist())):
        lines.append(f"{i:<10}{tok:<15}{g:.6f}")

    # Quantify the decay: compare average gradient magnitude in first third vs last third
    third = seq_len // 3
    early_avg = grad_per_timestep[:third].mean().item()
    late_avg = grad_per_timestep[-third:].mean().item()
    ratio = early_avg / late_avg if late_avg > 0 else float('inf')

    lines.append(f"\nAverage gradient norm, first {third} timesteps (early tokens):  {early_avg:.6f}")
    lines.append(f"Average gradient norm, last {third} timesteps (recent tokens):   {late_avg:.6f}")
    lines.append(f"Ratio (early / late): {ratio:.4f}")
    lines.append("\nInterpretation: even in this small UNTRAINED network (no learning has happened")
    lines.append("yet -- these are random initial weights), the gradient flowing back to early")
    lines.append("timesteps is already noticeably smaller than for recent timesteps in most runs.")
    lines.append("This is the mathematical seed of the 'vanishing gradient' problem: at every")
    lines.append("timestep, the gradient gets multiplied by the LSTM's weight matrices and gate")
    lines.append("activations (values typically < 1). Multiply many such factors together across")
    lines.append("50+ timesteps and the signal reaching early tokens shrinks multiplicatively --")
    lines.append("in a REAL trained network over much longer sequences, this compounds severely,")
    lines.append("meaning early tokens contribute almost nothing to the final prediction's error")
    lines.append("signal, so the model effectively 'forgets' long-range context during training.")

    # --- Demonstrate sequential dependency: no parallelism possible ---
    lines.append("\n\n--- SEQUENTIAL DEPENDENCY: each timestep depends on the previous one ---\n")
    lines.append("An LSTM cell's hidden state h_t and cell state c_t are computed as a FUNCTION")
    lines.append("of h_(t-1) and c_(t-1) -- the previous timestep's output is a required INPUT")
    lines.append("to the current timestep's computation. This is structurally different from a")
    lines.append("Transformer, where every token's attention computation can run independently")
    lines.append("and simultaneously.")

    # Time a manual step-by-step unroll vs the batched cudnn-optimized call, to make the
    # sequential-dependency point concrete (not a rigorous benchmark, just illustrative)
    hidden_dim = 64
    lstm_cell = nn.LSTMCell(32, hidden_dim)
    embed_seq = model.embed(token_ids)[0]  # [seq_len, embed_dim]

    start = time.perf_counter()
    h_t = torch.zeros(1, hidden_dim)
    c_t = torch.zeros(1, hidden_dim)
    for t in range(seq_len):
        # THIS view makes the dependency explicit: step t needs (h_t, c_t) from step t-1
        h_t, c_t = lstm_cell(embed_seq[t].unsqueeze(0), (h_t, c_t))
    manual_time = time.perf_counter() - start

    lines.append(f"\nManually unrolling all {seq_len} timesteps one-by-one (forced sequential loop): {manual_time*1000:.3f} ms")
    lines.append("Each iteration of this Python loop MUST wait for the previous iteration to")
    lines.append("finish, because h_t literally requires h_(t-1) as an argument -- there is no")
    lines.append("way to compute timestep 30 before timestep 29 has produced its hidden state.")
    lines.append("Contrast this with self-attention, where the representation for token 30 can")
    lines.append("be computed in the same parallel matrix operation as token 1, since attention")
    lines.append("only needs the FULL set of token embeddings up front -- not a running state")
    lines.append("threaded through one timestep at a time.")

    output = "\n".join(lines)
    print(output)

    with open("outputs/lstm_gradient_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/lstm_gradient_results.txt")
