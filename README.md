# AI/ML Internship Portfolio

This repository contains all work completed during my AI/ML internship, organized by day. Each day builds on the previous one, tracing the progression from classical NLP through modern context-sensitive representations.

## Structure

| Day | Focus | Folder |
|---|---|---|
| Day 1 | Classical NLP preprocessing pipeline — text cleaning, tokenization comparison, stemming/lemmatization, POS tagging, Bag-of-Words | [`day-1-nlp-pipeline/`](./day-1-nlp-pipeline) |
| Day 2 | From counting to predicting — TF-IDF manual verification, cosine retrieval, synonym-blindness proof, Word2Vec skip-gram, polysemy failure demonstration | [`day-2-tfidf-word2vec/`](./day-2-tfidf-word2vec) |
| Day 3 | From static vectors to context — N-gram language models, LSTM gradient analysis, BERT contextual embeddings, t-SNE visualization | [`day-3-context-attention/`](./day-3-context-attention) |

## The Full Arc

Each day is designed to expose a specific limitation of the previous day's approach:

- **Day 1** builds classical, purely lexical representations that treat text as strings — no meaning captured.
- **Day 2** replaces counting with predictive training (Word2Vec), demonstrably improving synonym detection but still producing one fixed vector per word — proven to fail on polysemy.
- **Day 3** introduces sequential models (n-grams, LSTMs), measures the vanishing-gradient problem directly, and uses pretrained BERT to resolve Day 2's open question by producing genuinely context-dependent embeddings.

The full technical progression is documented in each day's `REPORT.md` and `README.md`.

## Author

**Fatima Azeem**
AI/ML Internship
