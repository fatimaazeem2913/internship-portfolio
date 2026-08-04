# Final Task: Complete Transformer Walkthrough

**Query:** "Who is the president of Pakistan?"

This traces every stage of a decoder-only Transformer's forward pass for this exact query, connecting each stage back to the components built and verified in Days 1-4, and showing precisely how the final hidden state becomes a probability distribution over the first output token.

---

## Stage 1: Tokenization

The raw string is split into subword tokens using a BPE-style tokenizer (Day 1's exact technique). A real tokenizer would likely produce something like:

```
"Who is the president of Pakistan?"
-> ["Who", " is", " the", " president", " of", " Pakistan", "?"]
-> token IDs: [15546, 318, 262, 4732, 286, 6280, 30]
```

**What's being computed here:** raw text is converted into a sequence of integers from a fixed vocabulary (~50,000-100,000 entries, Day 4). Note "Pakistan" survives as a single token because it's common enough in training data to have earned its own vocabulary slot; a rarer proper noun might fragment into subword pieces (Day 1's BPE demonstration showed exactly this behavior). These integers carry no meaning yet -- they are pointers into a lookup table, nothing more (Day 4's tokens-vs-embeddings distinction).

---

## Stage 2: Token Embeddings

Each token ID indexes into a learned embedding matrix of shape [vocab_size, d_model] (e.g., [50257, 768] for a GPT-2-scale model):

```
15546 ("Who")           -> [0.021, -0.318, ..., 0.107]   (768 floats)
318   (" is")            -> [-0.142, 0.556, ..., -0.033]
262   (" the")            -> [0.089, -0.201, ..., 0.445]
4732  (" president")       -> [0.301, 0.112, ..., -0.276]
286   (" of")               -> [-0.055, 0.398, ..., 0.019]
6280  (" Pakistan")          -> [0.412, -0.087, ..., 0.334]
30    ("?")                   -> [-0.201, 0.150, ..., -0.098]
```

**What's being computed:** each token's arbitrary integer ID is converted into a dense, continuous vector whose position in space encodes learned meaning (Day 4). Critically, at this exact moment, "Pakistan"'s vector is static -- the same 768 numbers regardless of what sentence it appears in (this is exactly the Word2Vec-style limitation measured in Day 2; the next stages are what make it context-sensitive).

---

## Stage 3: Positional Encoding

A positional vector (sinusoidal, Day 4's formula, or a learned positional embedding as in GPT-2) is added to each token's embedding, one per position:

```
final_input[0] = embed("Who")         + PE(position=0)
final_input[1] = embed(" is")          + PE(position=1)
final_input[2] = embed(" the")          + PE(position=2)
final_input[3] = embed(" president")     + PE(position=3)
final_input[4] = embed(" of")             + PE(position=4)
final_input[5] = embed(" Pakistan")        + PE(position=5)
final_input[6] = embed("?")                 + PE(position=6)
```

**What's being computed:** self-attention (Stage 5) is inherently permutation-invariant -- it has no built-in sense of order (Day 4's verified proof: shuffling input without positional encoding produced identical attention output). Adding a position-dependent vector to each token breaks this symmetry, so "Pakistan" at position 5 is now numerically distinguishable from "Pakistan" appearing at position 0 in some other sentence. This is the last step before the sequence enters the actual Transformer layers.

---

## Stage 4: Q/K/V Creation

Within each attention layer, every token's current vector x_i is projected through three learned weight matrices:

```
Q_i = x_i @ W_Q     (What is this token looking for?)
K_i = x_i @ W_K     (What does this token contain?)
V_i = x_i @ W_V     (What does this token contribute if attended to?)
```

Concretely, for the token " president" (position 3):

```
Q_3 = x_3 @ W_Q   ->  a 64-dimensional vector (assuming 12 heads, d_model=768 -> 64 per head)
K_3 = x_3 @ W_K   ->  a 64-dimensional vector
V_3 = x_3 @ W_V   ->  a 64-dimensional vector
```

This happens for every token, in parallel (Day 4's key architectural point -- no sequential dependency here, unlike Day 3's LSTM hidden state). With multi-head attention (Day 4), this whole process repeats independently across, say, 12 heads, each working on a 64-dimensional slice, allowing different heads to specialize in different relationship types.

---

## Stage 5: Masked Self-Attention

Every token's Query is compared against every earlier-or-equal token's Key (the causal mask, Day 4 -- critical for a decoder-only, generative model):

```
scores = Q @ K^T / sqrt(d_k)                # raw compatibility scores, scaled
scores_masked = apply_causal_mask(scores)   # future positions set to -inf
attention_weights = softmax(scores_masked)  # each row sums to 1.0
output = attention_weights @ V              # weighted sum of Value vectors
```

Concretely, for the final token "?" (position 6), whose attention output matters most since it's the token the next prediction will be generated from:

```
"?" attends to: "Who"(0), " is"(1), " the"(2), " president"(3), " of"(4), " Pakistan"(5), "?"(6)
```

A well-trained model's attention weights here would likely be heavily concentrated on " president" and " Pakistan" -- these are the semantically load-bearing tokens for answering the question, while "Who", " is", " the", " of" carry mostly grammatical/structural information and would typically receive lower (though non-zero) attention weight. The causal mask ensures "?" cannot attend to any token that would come after it (there are none yet, since this is the last token of the input, but the mechanism is what prevents any token from "peeking ahead" during training on longer sequences).

**What's being computed:** this is the stage where tokens genuinely communicate. "?"'s new representation is no longer just "a question mark" -- it now carries a blended signal that says, in effect, "this sequence is asking about the identity of a president, specifically Pakistan's president." This is exactly the mechanism Day 3 measured indirectly (BERT producing different vectors for "light" in different contexts) -- attention is why that happens.

**Multi-head note:** with 12 heads, this whole computation happens 12 times in parallel over different 64-dim subspaces, then the 12 outputs are concatenated back to 768 dimensions and passed through an output projection W_O (Day 4).

---

## Stage 6: Feed-Forward Network (FFN)

Each token's post-attention representation is passed through a position-wise FFN, independently (Day 4 -- verified no cross-token mixing here; all cross-token communication already happened in Stage 5):

```
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2      # 768 -> 3072 -> 768
```

**What's being computed:** the FFN is where much of the model's factual/associative "knowledge" is believed to live (Day 4's key-value-memory framing). For the "?" token's now attention-enriched representation (which encodes "asking about Pakistan's president"), the FFN's learned weights can inject associative information -- for instance, if the training data strongly associated "Pakistan" + "president" with "Zardari," this is a plausible stage where that association gets reinforced into the representation, even though the FFN operates on this one token's vector in isolation, with no direct access to the other tokens (it relies entirely on what attention already blended in).

---

## Stage 7: Residual Connections + Layer Normalization (applied around both Stage 5 and Stage 6)

```
x = x + Attention(LayerNorm(x))      # Pre-Norm, modern convention (Day 4)
x = x + FFN(LayerNorm(x))
```

**What's being computed:** the residual "+x" ensures the gradient (during training) and the original signal (during inference) are never lost, even as the representation passes through dozens of these blocks (Day 4's measured contrast: without residuals, signal decays exponentially; Day 3's LSTM showed the real-world cost of this decay when it isn't architecturally prevented). LayerNorm keeps the numbers in a stable, consistent range at every step. This entire Stage 4-7 sequence is one Transformer layer; GPT-2 Small repeats it 12 times, larger models more.

---

## Stage 8: Hidden States (After All N Layers)

After passing through all N layers, every token position has a final hidden state -- a 768-dimensional vector. Only the LAST token's hidden state matters for predicting the next token (in a decoder-only model generating left-to-right):

```
final_hidden_state["?"] = [a rich 768-dim vector that has, by this point,
                            been shaped by attention over "Who/is/the/
                            president/of/Pakistan/?" across 12 layers,
                            and by 12 rounds of FFN knowledge injection]
```

**What's being computed:** this single vector is the model's complete "understanding," at this point in the network, of the entire input sequence, concentrated into the position it needs to predict from next. Every earlier token's information that was relevant has, in principle, been propagated into this vector through the repeated attention operations across all N layers.

---

## Stage 9: Vocabulary Projection

The final hidden state is projected from d_model (768) dimensions to vocab_size (~50,257) dimensions via a linear layer (often weight-tied with the token embedding matrix from Stage 2, reusing the same parameters in reverse):

```
logits = final_hidden_state["?"] @ W_vocab      # [768] @ [768, 50257] -> [50257]
```

**What's being computed:** one raw, unnormalized score for every single token in the entire vocabulary -- a number representing "how compatible is this token with everything the model has computed so far?" At this stage, the scores are NOT yet probabilities -- they can be any real number, positive or negative.

---

## Stage 10: Softmax -> Probability Distribution

```
probabilities = softmax(logits)      # Day 4's exact formula
```

Illustrative (not real-model) probabilities for this query:

```
P("Asif")        = 0.41   <- highest, start of the correct name
P("The")         = 0.18   <- e.g. "The current president is..."
P("Zardari")     = 0.09
P("As")          = 0.07
P("A")           = 0.03
...
P("Elephant")    = 0.0000001   <- vanishingly unlikely, but never exactly zero
```

**What's being computed:** the raw logits are converted into a valid probability distribution -- every value in [0,1], summing to exactly 1.0 (Day 4's verified property). The token with the highest probability ("Asif," beginning the correct answer "Asif Ali Zardari," per this project's own retrieved context) reflects that the model's entire preceding computation -- tokenization, embedding, positional encoding, twelve (or more) layers of attention and FFN -- has converged on this being the single most probable continuation.

---

## Generating the First Output Token

A token is sampled from this distribution (Day 5/6: greedy = always pick the argmax, i.e. "Asif"; or via temperature/top-k/top-p sampling for more varied output). Say "Asif" is selected. Critically, per Day 5's autoregressive mechanism: this selected token is now appended to the sequence, and the entire Stage 1-10 pipeline repeats -- "Asif" becomes a new input token, gets embedded, gets positional-encoded, and the model computes a fresh probability distribution for the next token (likely "Ali"), conditioned now on "...Pakistan? Asif" rather than just "...Pakistan?". This repeats one token at a time until the model produces a complete answer.

---

## The Full Arc, in One Sentence

Tokenization turns text into pointers; embeddings turn pointers into static meaning; positional encoding adds order; Q/K/V creation prepares each token to ask questions and offer information; masked self-attention lets tokens exchange information about "president" and "Pakistan" specifically with the query token; the FFN injects learned associative knowledge; residuals and LayerNorm keep this stable across many stacked layers; the final hidden state of the last token is a complete, context-saturated summary of the whole input; and the vocabulary projection plus softmax converts that summary into a genuine probability distribution -- from which the very first token of the answer, "Asif," is sampled, kicking off the autoregressive loop that produces the rest of the response one token at a time.
