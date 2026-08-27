# Simple RAG Failure Modes & Critical Evaluation

## Executive Summary
While Simple (Naive) RAG succeeds on direct, single-paragraph factual lookups, it exhibits predictable failure modes when applied to complex production documents. Below is a systematic analysis of failure modes observed across retrieval, context assembly, and synthesis.

---

## 1. Retrieval Failures (Garbage In, Garbage Out)

### A. Vocabulary Mismatch (Dense Vector Blindspots)
* **Failure Mechanism**: Dense bi-encoders embed semantic concepts but can miss exact keyword matches for domain codes, legal clause IDs, acronyms, or exact product numbers (e.g., `ERROR_403_CSRF` vs `forbidden request`).
* **Symptom**: Low cosine similarity ranking for queries requiring lexical precision.
* **Remedy (Day 18+ Transition)**: **Hybrid Search** (Dense Cosine + Sparse BM25 / Reciprocal Rank Fusion).

### B. Out-of-Domain / Unanswerable Queries (False Positives)
* **Failure Mechanism**: Standard top-k nearest neighbor search always returns the mathematical top-k closest vectors, even if cosine distance is high (>0.70).
* **Symptom**: Irrelevant chunks are injected into the prompt context.
* **Remedy**: Calibrated **Confidence Score Thresholding** (e.g., discard chunks with similarity score <0.45).

---

## 2. Context Assembly & Splitting Failures

### A. Context Fragmentation ("Lost in the Middle")
* **Failure Mechanism**: Fixed-token splitting cuts an explanation or formula across two chunks.
* **Symptom**: The retriever returns Chunk A with the premise, but Chunk B with the conclusion is excluded.
* **Remedy**: **Hierarchical Chunking** (Parent-Child indexing) and **Context Window Expansion**.

### B. Multi-Hop / Cross-Document Disconnection
* **Failure Mechanism**: Queries requiring information from two separate manuals or chapters fail if the query vector aligns with only one topic.
* **Symptom**: Incomplete synthesis.
* **Remedy**: **Multi-Query Decomposition** and **Hypothetical Document Embeddings (HyDE)**.

---

## 3. Generation & LLM Failures

### A. Extrapolation & Prior Knowledge Hallucination
* **Failure Mechanism**: LLMs have strong internal priors and tend to answer ungrounded questions from parametric memory instead of saying "I do not know."
* **Remedy**: Strict System Instructions enforcing bracketed source citations `[Source: X, Page: Y]` and zero-temperature decoding.