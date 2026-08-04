# Week 1 Learning Log

## What Surprised Me

**How dramatic and clean the "proof" moments were.** I expected the TF-IDF synonym-blindness result (Day 2) to be *low*, not *exactly zero*. Seeing 0.0000 across all 5 synonym pairs made the limitation feel undeniable rather than approximate — it wasn't "TF-IDF is bad at synonyms," it was "TF-IDF is mathematically incapable of ever scoring these above zero, by construction." The same surprise happened with the LSTM gradient measurement in Day 3 — I expected "the gradient gets smaller," not literally `0.000000` at the earliest timesteps of an *untrained* network. I had assumed vanishing gradients were something that emerged gradually during bad training, not something baked into the architecture from the very first forward pass.

**How much of "prompt engineering" turned out to be about disambiguation, not persuasion.** Before Day 6, I think I imagined prompt engineering as finding the right "magic words" to unlock better performance. Actually building the zero-shot vs. few-shot experiments made it obvious that the model almost never lacked the underlying knowledge — it lacked information about *which* of several reasonable interpretations I wanted. That reframing (prompting as disambiguation, not persuasion) changed how I think about every prompt I write now.

**How small the actual architectural difference is between an "understanding" model and a "generating" model.** I expected BERT and GPT to be fundamentally different systems. Building the same Transformer block in Day 4 and toggling a single causal mask on and off to switch between encoder-style and decoder-style behavior made this concrete in a way reading about it never did — it's the same weights, same FFN, same LayerNorm, one boolean difference in the attention mask.

## What Clicked Immediately

**The Q/K/V analogy, once I traced real numbers through it.** The library-search analogy for attention (Query = your question, Key = each book's label, Value = the book's content) made intuitive sense the first time I read about it, but it didn't fully click until I hand-verified the actual softmax weights in Day 4 and saw [0.4011, 0.1978, 0.4011] come out of real dot products. After that, reading any attention diagram became instantly legible.

**Why residual connections and LayerNorm have to come as a pair.** The moment I actually measured that residuals alone cause gradients to grow exponentially (not just "not vanish"), the reason LayerNorm always appears right next to a residual connection in every architecture diagram stopped being an arbitrary convention and became an obvious necessity.

**The autoregressive loop itself.** "Predict a distribution, sample, append, repeat" is a genuinely simple idea, and reusing my own working trigram model in Day 5 to walk through it step-by-step made it click faster than any amount of reading about GPT would have, because I could see the actual probabilities at each step, not just the concept.

## What Still Needs Deliberate Practice

**Building real intuition for when embeddings will fail, not just when they succeed.** I can explain why Word2Vec fails at polysemy in the abstract, but I don't yet have a fast, reliable instinct for predicting in advance whether a given real-world retrieval task will hit that wall before I've actually run the experiment. That instinct seems to come from seeing many more failure cases across different domains than I've been exposed to in one week.

**Estimating the real cost/latency tradeoffs in production without running the numbers.** I understand conceptually why CoT, few-shot examples, and larger context windows all cost more, but I don't yet have a gut sense of how much more in a real production system. I'd need to actually price out a few real API workloads to build that intuition rather than just knowing the mechanism exists.

**Debugging environment/infrastructure issues faster.** The PyTorch CUDA-dependency crisis in Day 3, the venv breaking after a folder move, the GitHub authentication back-and-forth. I got through all of them, but slower than I'd like. I suspect this genuinely just needs more repetition; the underlying concepts (dependency resolution, environment isolation, SSH auth) are things I understand, but recognizing the specific failure signature quickly under time pressure is a different skill than understanding the concept.

**Holding the full Transformer forward pass in my head all at once.** I can explain each stage (tokenize, embed, add position, attend, FFN, project, softmax) individually and confidently, but tracing the entire pipeline for a specific real query end-to-end, the way the Day 7 final task requires, still takes deliberate, careful step-by-step work rather than being something I can do fluently from memory. This is very likely just a matter of repetition — doing this walkthrough for several different queries would probably make it automatic.
