# AI/ML Internship Portfolio

This repository contains all work completed during my AI/ML internship, organized by day. Each day builds on the previous one, tracing the progression from classical NLP through modern context-sensitive representations.

## Structure

| Day | Focus | Folder |
|---|---|---|
| Day 1 | Classical NLP preprocessing pipeline — text cleaning, tokenization comparison, stemming/lemmatization, POS tagging, Bag-of-Words | [`day-1-nlp-pipeline/`](./day-1-nlp-pipeline) |
| Day 2 | From counting to predicting — TF-IDF manual verification, cosine retrieval, synonym-blindness proof, Word2Vec skip-gram, polysemy failure demonstration | [`day-2-tfidf-word2vec/`](./day-2-tfidf-word2vec) |
| Day 3 | From static vectors to context — N-gram language models, LSTM gradient analysis, BERT contextual embeddings, t-SNE visualization | [`day-3-context-attention/`](./day-3-context-attention) |
| Day 4 | Transformer architecture — attention from scratch in NumPy, positional encoding, full block implementation, GPT-2 decoder-only study | [`day-4-transformer-architecture/`](./day-4-transformer-architecture) |
| Day 5 | How LLMs work internally — training pipeline, autoregressive generation, sampling strategies from scratch, architecture comparison, emergent capabilities | [`day-5-llm-internals/`](./day-5-llm-internals) |
## The Full Arc

Each day is designed to expose a specific limitation of the previous day's approach:

- **Day 1** builds classical, purely lexical representations that treat text as strings — no meaning captured.
- **Day 2** replaces counting with predictive training (Word2Vec), demonstrably improving synonym detection but still producing one fixed vector per word — proven to fail on polysemy.
- **Day 3** introduces sequential models (n-grams, LSTMs), measures the vanishing-gradient problem directly, and uses pretrained BERT to resolve Day 2's open question by producing genuinely context-dependent embeddings.
- **Day 4** builds the Transformer from first principles in pure NumPy, showing how one architecture resolves every limitation measured in Days 1–3 — and why the O(n²) cost of attention became the defining constraint of the LLM era.
The full technical progression is documented in each day's `REPORT.md` and `README.md`.
- **Day 5** shows how these components are assembled into a real training pipeline (pre-training → SFT → RLHF), demonstrates that autoregressive generation is just "predict, sample, append, repeat" using a real self-built language model, and researches the emergent capabilities that appear only once this simple objective is scaled far enough.
## Author

**Fatima Azeem**
AI/ML Internship
