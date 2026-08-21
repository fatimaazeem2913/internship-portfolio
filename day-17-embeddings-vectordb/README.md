# Embedding Models & Vector Databases — Day 17 Internship

## Project Overview

This project was completed as part of the Day 17 internship tasks under
**Phase 3: Production RAG Systems**. The objective was to evaluate,
configure, and benchmark embedding models and vector database
architectures (ChromaDB vs. FAISS) on **1,140 real document chunks**
ingested from Day 16, measuring speed-versus-quality trade-offs and
verifying 100% metadata lineage.

The benchmark engine embeds every chunk with each of 3 real embedding
models, indexes the resulting vectors in both ChromaDB and FAISS, runs
20 real benchmark questions against every model × store combination,
and calculates individual search latency in milliseconds and Top-3
Precision (P@3) for each.


## Objectives

* Benchmark and compare three dense embedding models: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, and `BAAI/bge-large-en-v1.5` on ingestion throughput and semantic fidelity[cite: 3, 10].
* Install and configure **ChromaDB**: build persistent collections, batch-embed 1,140 document chunks, and store them alongside full metadata lineage[cite: 3, 10, 11].
* Install and configure **FAISS**: implement an identical high-speed ingestion pipeline and compare similarity search query latency against ChromaDB[cite: 3, 10, 11].
* Implement vector database collection management utilities (CRUD): list, count, delete, and incremental insertion[cite: 3, 11].
* Execute an automated retrieval benchmark with 20 factual ground-truth questions, measuring **Top-3 Precision (P@3)** for each model and vector store combination[cite: 3, 8, 10].
* Document technical trade-offs to establish clear production deployment recommendations[cite: 3, 6].

---

## Technologies Used

* **Sentence-Transformers & HuggingFace Hub** (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `BAAI/bge-large-en-v1.5`)[cite: 3, 10, 12]
* **ChromaDB 0.4+** (Persistent client, SQLite relational metadata catalog, HNSW C++ graph index)[cite: 5, 11]
* **FAISS-CPU** (In-memory C++ vectorized `IndexFlatIP` inner product / cosine similarity)[cite: 5, 6, 11]
* **NumPy** (Dense matrix normalization and binary `.npy` array persistence)[cite: 5, 11]
* **Pytest** (Automated pipeline and metadata lineage test suite)[cite: 5, 7]
* **Tabulate** (Terminal CLI benchmark matrix formatting)[cite: 5, 10]

---

## Project Structure

```text
day-17-embeddings-vectordb/
├── data/
│   ├── benchmark_questions.json          # 20 ground-truth questions with expected keyword targets
│   └── chunks_hierarchical.json          # 1,140 hierarchical chunks from Day 16 ingestion
├── docs/
│   └── TRADEOFF_ANALYSIS.md              # Detailed speed, latency, and precision trade-off report
├── outputs/
│   ├── chroma_db/                        # ChromaDB persistent collections
│   │   ├── <collection_uuid_1>/          # HNSW binary graph indexes (data_level0.bin, etc.)
│   │   ├── <collection_uuid_2>/          
│   │   ├── <collection_uuid_3>/          
│   │   └── chroma.sqlite3                # Relational SQLite metadata store
│   ├── benchmark_results.json            # Exported quantitative benchmark metrics
│   ├── chunks_with_embeddings.json       # Document chunks enriched with 384d vector coordinates
│   └── embeddings_matrix.npy             # Raw (1140, 384) NumPy float32 matrix
├── src/
│   ├── __init__.py
│   ├── benchmark.py                      # Multi-model evaluation and Top-3 Precision engine
│   ├── embeddings.py                     # Embedding wrapper for MiniLM, MPNet, and BGE-Large
│   └── vector_stores.py                  # ChromaStoreManager & FAISSStoreManager CRUD engines
├── tests/
│   ├── __init__.py
│   └── test_vector_pipeline.py           # Pytest test suite (embeddings, FAISS, metadata lineage)
├── main.py                               # Master benchmark execution runner
├── README.md                             # Project documentation
└── requirements.txt                      # Project dependencies


## Tasks Performed

### 1. Multi-Model Embedding Adapter Layer
Constructed `EmbeddingModelWrapper` in `src/embeddings.py` supporting dimension scaling across 384d, 768d, and 1024d with batch embedding, vector normalization, and execution timers.

### 2. ChromaDB Persistent Architecture & Metadata Lineage
Implemented `ChromaStoreManager` in `src/vector_stores.py` using `chromadb.PersistentClient`. Stored all 1,140 chunks alongside structured metadata tags (`source`, `page_number`, `chunk_index`, `section_heading`) enabling metadata-filtered queries (`where={"source": "..."}`).

### 3. FAISS High-Speed Ingestion & Search Engine
Built `FAISSStoreManager` utilizing `faiss.IndexFlatIP` on unit-normalized vectors for exact cosine similarity search, maintaining parallel document and metadata lookups.

### 4. Vector Store CRUD & Collection Utilities
Added and verified administrative methods: `add_documents()`, `search()`, `count()`, and `delete_collection()` across both stores.

### 5. Automated Retrieval Benchmark Engine
Engineered `src/benchmark.py` to evaluate 20 factual questions across all model-store combinations, calculating individual search latency in milliseconds and Top-3 Precision ($P@3$).

---


## Results & Benchmark Metrics

```
================================================================
 DAY 17: EMBEDDINGS & VECTOR DATABASE BENCHMARK ENGINE
 Ingested Chunks: 1,140 | Benchmark Questions: 20
================================================================
```

| Embedding Model | Dimension | Ingestion Time (1,140 Chunks) | Throughput | ChromaDB Top-3 Precision | ChromaDB Query Latency | FAISS Top-3 Precision | FAISS Query Latency |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 0.82s | 1,390 chunk/s | 83.3% | 2.84 ms | 83.3% | 0.14 ms |
| all-mpnet-base-v2 | 768 | 2.45s | 465 chunk/s | 90.0% | 3.21 ms | 90.0% | 0.22 ms |
| BAAI/bge-large-en-v1.5 | 1024 | 6.10s | 186 chunk/s | 96.7% | 3.95 ms | 96.7% | 0.28 ms |

- **3/3 Pytest unit tests passed** (`test_embedding_wrapper`,
  `test_faiss_vector_store`, `test_metadata_lineage_preservation`) with
  zero collection errors.
- **100% metadata lineage maintained** across all 1,140 chunks, with
  zero orphan vectors.

## Observations

- **Speed vs. dimension scaling** — embedding latency scales linearly
  with parameter count and vector dimensionality. `all-MiniLM-L6-v2` is
  **7.4x faster** than `bge-large-en-v1.5`, making it an effective choice
  for throughput-limited environments.
- **Retrieval quality under technical complexity** — on complex formulas
  (e.g. MSE loss formulations, token bucket rates, and legal clauses),
  `BAAI/bge-large-en-v1.5` achieved the highest accuracy (**96.7% Top-3
  Precision**), successfully capturing domain-specific terminology that
  smaller models missed.
- **Vector store latency disparity** — FAISS demonstrated **14x-20x
  faster search speeds** (0.14 ms vs. 2.84 ms) due to its bare-metal C++
  in-memory index. ChromaDB introduces a slight SQLite metadata query
  overhead but provides immediate persistence and metadata filtering out
  of the box.

## Challenges Encountered

- **Handling multi-engine metadata synchronization in FAISS** — FAISS
  only indexes raw floating-point arrays, with no native metadata
  storage capability. To match ChromaDB's metadata capabilities, an
  internal ID-to-metadata registry was implemented in
  `FAISSStoreManager` to return matching document metadata upon
  retrieval.
- **Vector normalization for cosine metric consistency** — by default,
  inner product search (`IndexFlatIP`) only reflects true cosine
  similarity when vectors have unit length. A strict L2-normalization
  step was added across all document and query embeddings before FAISS
  index insertion.

## How to Run

### 1. Setup Environment

```bash
# Navigate to project folder
cd day-17-embeddings-vectordb

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Benchmark Engine

```bash
python main.py
```

### 3. Run the Automated Test Suite

```bash
python -m pytest tests/ -v
```

## Learning Outcomes

- How to benchmark dense embedding models across dimensionality (384d
  vs. 768d vs. 1024d) and evaluate their real-world impact on precision
  and ingestion throughput.
- How ChromaDB structures persistent vector storage by separating
  relational metadata (SQLite) from HNSW graph binaries (`hnswlib`).
- How to configure FAISS for normalized cosine similarity search
  (`IndexFlatIP`) and pair it with memory lookups for metadata lineage.
- How to design an objective, keyword-grounded factual retrieval
  evaluation pipeline using Top-3 Precision.

## Author

**Fatima Azeem** — AI/ML Internship (Phase 3: Production RAG Systems, Day 17)