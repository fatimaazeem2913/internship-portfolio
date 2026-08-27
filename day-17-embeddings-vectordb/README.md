# Embedding Models & Dense Vector Retrieval Benchmark — Day 17

## Project Overview

This module benchmarks dense vector representation pipelines and vector
indexing backends for technical Retrieval-Augmented Generation (RAG). It
evaluates trade-offs across embedding dimensionalities (384-d, 768-d,
and 1024-d) and compares vector store architectures (**ChromaDB** vs.
**FAISS**) on ingestion throughput, retrieval precision (P@3), metadata
filtering, and query latency across technical documents, mathematical
formulations, and multimodal visual descriptions.

## Objectives

- **Dimensionality benchmarking** — evaluate performance trade-offs
  among embedding models across 384-d (`all-MiniLM-L6-v2`), 768-d
  (`all-mpnet-base-v2`), and 1024-d (`BAAI/bge-large-en-v1.5`).
- **Vector store comparison** — implement and compare **ChromaDB**
  (persistent, metadata-aware) against **FAISS** (in-memory flat
  `IndexFlatIP`).
- **Information retrieval evaluation** — measure Precision@3 (P@3) and
  query retrieval latency across technical curriculum queries and
  benchmark question sets.
- **Architecture selection** — determine the optimal model and database
  configuration for the downstream hybrid retrieval system.

## Technologies Used

- **Embedding Models:** HuggingFace / SentenceTransformers
  (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `BAAI/bge-large-en-v1.5`)
- **Vector Databases:** ChromaDB (`chromadb`), Facebook AI Similarity
  Search (`faiss-cpu`)
- **Core Runtime & Scaffolding:** Python 3.12, NumPy, Rich, Tabulate,
  Pytest

## Project Structure

```text
day-17-embeddings-vectordb/
├── data/
│   ├── benchmark_questions.json
│   └── chunks_hierarchical.json
├── docs/
│   └── TRADEOFF_ANALYSIS.md
├── outputs/
│   ├── chroma_db/
│   │   └── chroma.sqlite3
│   ├── embeddings_matrix.npy
│   └── benchmark_results.json
├── src/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── embeddings.py
│   └── vector_stores.py
├── tests/
│   └── test_vector_pipeline.py
├── main.py
├── requirements.txt
└── README.md
```

## Tasks Performed

- Built a standardized embedding abstraction layer (`embeddings.py`)
  supporting normalized dense embeddings with dimension validation.
- Implemented modular vector database managers (`vector_stores.py`):
  - `ChromaStoreManager` — SQLite-backed persistent collection storage
    with document, metadata, and distance querying.
  - `FAISSStoreManager` — in-memory `IndexFlatIP` cosine inner-product
    indexing with vector normalization.
- Executed an automated evaluation harness (`benchmark.py`, `main.py`)
  benchmarking 1,140 ingested passages across 20 specialized technical
  validation queries.
- Saved benchmark results to `outputs/benchmark_results.json` and
  generated trade-off analysis documentation in
  `docs/TRADEOFF_ANALYSIS.md`.

## Empirical Benchmark Results

| Model | Dimensions | Ingestion Time | Throughput | Chroma P@3 | Chroma Latency | FAISS P@3 | FAISS Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `all-MiniLM-L6-v2` | 384 | 7.73 s | 147.5 chunks/s | 98.3% | 1.42 ms | 53.0% | < 0.1 ms |
| `all-mpnet-base-v2` | 768 | 59.23 s | 19.2 chunks/s | 98.3% | 1.54 ms | 60.0% | < 0.1 ms |
| `BAAI/bge-large-en-v1.5` | 1024 | 189.05 s | 6.0 chunks/s | **98.3%** | 1.99 ms | 67.0% | < 0.1 ms |

## How to Run

```bash
# 1. Activate virtual environment
source ../day-19-hybrid-search-advanced-retrieval/venv/bin/activate

# 2. Run ingestion & benchmark pipeline
python main.py

# 3. Run test suite
python -m pytest tests/
```

## Learning Outcomes

- Evaluated speed, throughput, and accuracy trade-offs across 384-d,
  768-d, and 1024-d embedding spaces.
- Built production wrappers for ChromaDB and FAISS with persistence and
  query interfaces.
- Validated vector pipeline performance for downstream hybrid retrieval
  integration.

## Author

**Fatima Azeem** — AI/ML Internship (Phase 3, Day 17)
