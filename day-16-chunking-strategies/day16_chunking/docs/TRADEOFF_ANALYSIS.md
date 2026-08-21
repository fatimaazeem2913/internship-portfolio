# Chunking & Retrieval Trade-Off Analysis

## 1. Chunk Size vs. Retrieval Precision
- **Small Chunks (100–250 tokens / 300–800 chars)**:
  - *Pros*: Minimizes vector noise; high cosine similarity precision for specific factual queries (e.g., specific API rate limits, error codes).
  - *Cons*: Context fragmentation; missing overarching semantic relationships.
- **Large Chunks (800–1500 tokens / 2500–5000 chars)**:
  - *Pros*: Preserves complete conceptual explanations, proofs, and multi-step algorithm workflows.
  - *Cons*: Vector embedding dilution; embedding averages out distinct semantic topics.

## 2. Overlap Optimization
- **Rule of Thumb**: 10% to 15% overlap.
- **Mechanism**: Eliminates "semantic cliffing" where named entities, equations, or boundary clauses get severed across chunk splits.

## 3. When Semantic Chunking Beats Fixed-Size Chunking
- **Semantic Chunking Wins**:
  - Legal agreements, regulatory compliance policies, and scientific textbook chapters where each paragraph represents an atomic thought.
- **Fixed-Size / Recursive Wins**:
  - Raw system logs, time-series tables, dense code files, and unstructured homogeneous transcripts.
