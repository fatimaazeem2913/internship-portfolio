# BM25, Hybrid Search & Advanced Retrieval — Day 19

## Project Overview

This project was completed as part of the Day 19 internship tasks under
Phase 3: Production RAG Systems. The objective was to build and
benchmark a production-grade hybrid retrieval architecture combining
sparse lexical search (BM25), dense vector search (ChromaDB), query
transformation, cross-encoder re-ranking, and hierarchical parent-window
context expansion.

Every milestone requirement is backed by verified implementation: **6/6
pytest tests passing**, a 20-question benchmark across 4 query
archetypes, an interactive multi-mode CLI comparison, and an empirical
evaluation report (`docs/HYBRID_RETRIEVAL_BENCHMARK.md`).

## Objectives

1. **BM25 sparse retrieval** — implement `BM25Okapi` with token
   normalization and parameter customization (k₁ = 1.5, b = 0.75).
2. **Hybrid search via RRF** — merge unbounded BM25 scores and dense
   cosine distance rankings using Reciprocal Rank Fusion (k = 60).
3. **Query rewriting** — leverage LLM query optimization to correct
   typos, resolve acronyms, and expand synonyms prior to search.
4. **Cross-encoder re-ranking** — deploy
   `cross-encoder/ms-marco-MiniLM-L-12-v2` over the top-20 retrieved
   candidates to compute full joint query-passage attention.
5. **Hierarchical context expansion** — retrieve fine-grained child
   chunks for high-precision vector similarity, then map back to parent
   document windows (600–1000 tokens) during generation.
6. **20-question multi-paradigm benchmark** — compare Simple Dense vs.
   BM25 Sparse vs. Hybrid (RRF) vs. Hybrid + Re-Rank across 20 diverse
   queries.

## Technologies Used

- **Rank-BM25** (`BM25Okapi` for lexical term frequency & document
  length normalization)
- **Sentence-Transformers** (`all-MiniLM-L6-v2` dense bi-encoder &
  `cross-encoder/ms-marco-MiniLM-L-12-v2`)
- **ChromaDB 0.4+** (persistent SQLite vector store with HNSW indexing)
- **Google Generative AI / GenAI SDK** (`gemini-3.6-flash` / offline
  test engine)
- **Rich & Tabulate** (interactive terminal formatting, tables, and
  telemetry)
- **Pytest** (automated 6/6 test verification suite)

## Project Structure

```text
day-19-hybrid-search-advanced-retrieval/
├── data/
│   ├── eval_20_questions.json          # 20-question multi-paradigm testbed
│   └── sample_corpus.json              # Multi-document parent-child chunk corpus
├── docs/
│   ├── HYBRID_RETRIEVAL_BENCHMARK.md   # Systematic 20-question comparison matrix
│   └── PROMPT_SPECIFICATION.md         # Grounded prompt and query rewriter contracts
├── outputs/
│   ├── chroma_db/                      # Persistent ChromaDB vector files
│   └── eval_20_benchmark_results.json  # Exported 20-question metric records
├── src/
│   ├── __init__.py
│   ├── bm25_retriever.py               # Sparse BM25Okapi implementation
│   ├── dense_retriever.py              # Dense ChromaDB vector retriever
│   ├── hybrid_search.py                # Reciprocal Rank Fusion (RRF, k=60)
│   ├── query_rewriter.py               # Query reformulation & typo corrector
│   ├── reranker.py                     # Cross-encoder (ms-marco-MiniLM-L-12-v2)
│   ├── hierarchical_manager.py         # Child-to-parent window context expander
│   └── pipeline_advanced.py            # Master orchestrator for all 4 search modes
├── tests/
│   ├── __init__.py
│   └── test_hybrid_rag.py              # 6/6 pytest verification test suite
├── main.py                             # Interactive CLI & 20-question benchmark
├── README.md
└── requirements.txt
```

## Tasks Performed & Technical Implementation

### 1. BM25 Sparse Lexical Search

Implemented in `src/bm25_retriever.py`. Tokenizes incoming queries and
calculates k₁-saturated term frequency and b-normalized document length
scores. Excels at exact keyword lookups and error codes
(`ERR_CONN_TIMEOUT_403`).

### 2. Reciprocal Rank Fusion (RRF)

Implemented in `src/hybrid_search.py`. Merges rankings without requiring
score normalization:

```
RRF_Score(d) = Σ  1 / (k + rank_m(d))       for each retrieval method m,  k = 60
```

### 3. Query Rewriting & Expansion

Implemented in `src/query_rewriter.py`. Uses `gemini-3.6-flash` (with a
deterministic rule-based fallback) to correct severe typographical noise
(`slinear reggression` → `linear regression`) and expand domain
acronyms (`CE loss` → `Cross Entropy log loss`).

### 4. Cross-Encoder Re-Ranking

Implemented in `src/reranker.py`. Deploys
`cross-encoder/ms-marco-MiniLM-L-12-v2` over the top-20 candidates from
hybrid search, performing full all-to-all cross-attention between query
and passage tokens.

### 5. Hierarchical Context Expansion

Implemented in `src/hierarchical_manager.py`. Uses child chunk vectors
for needle-in-a-haystack similarity matching in ChromaDB, then resolves
`parent_id` to expand prompt context to the complete parent section.

## 20-Question Benchmark Results Summary

```text
================================================================================
 DAY 19: 20-QUESTION SYSTEMATIC BENCHMARK ACROSS 4 RETRIEVAL PARADIGMS
================================================================================
```

| Evaluation Metric | Simple Dense (Day 18) | BM25 Sparse | Hybrid Search (RRF) | Hybrid + Re-Rank (Top-20) |
|---|:---:|:---:|:---:|:---:|
| **Exact Keyword Recall@3** | 68.0% | **98.0%** | 98.0% | **100.0%** |
| **Semantic / Concept Recall@3** | **98.0%** | 61.0% | 98.0% | **100.0%** |
| **Typo & Acronym Resilience** | 45.0% | 17.5% | 97.5% | **100.0%** |
| **Mean Average Precision (MAP)** | 0.73 | 0.69 | 0.91 | **0.98** |
| **Average Retrieval Latency** | 18.4 ms | **2.1 ms** | 22.8 ms | 48.6 ms |
| **Hallucination on Negative Tests** | 0.0% | 0.0% | 0.0% | **0.0% (100% Refusal)** |

## Observations & Engineering Insights

- **BM25 vs. dense synergy** — BM25 handles exact identifier codes,
  acronyms, and specific formula names where dense bi-encoders suffer
  from semantic dispersion, reflected directly in the 98.0% vs. 61.0%
  Exact Keyword Recall@3 gap above.
- **Query rewriting efficacy** — rewriting vague or typo-laden prompts
  into explicit keyword-rich queries drives the jump from 45.0% to
  97.5%+ typo and acronym resilience once hybrid retrieval is engaged.
- **Cross-encoder precision** — re-ranking effectively filters out
  false-positive dense vector matches, pushing every recall metric to
  100.0% at the cost of the highest latency in the pipeline (48.6 ms).

## Challenges Encountered

- **Score scale incompatibility** — raw BM25 scores are unbounded
  positive floats, whereas cosine similarities range from 0 to 1. Using
  rank-based Reciprocal Rank Fusion (RRF) bypassed the need for
  empirical score normalization.
- **Latency vs. accuracy trade-off** — cross-encoder inference increases
  CPU processing time; candidate pools were capped at the top-20 hybrid
  results prior to re-ranking to keep this trade-off manageable
  (48.6 ms average, still well within an interactive budget).

## How to Run

### 1. Environment Setup

```bash
# Navigate to project directory
cd day-19-hybrid-search-advanced-retrieval

# Activate virtual environment and install dependencies
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite (6/6 tests)

```bash
python -m pytest tests/ -v
```

### 3. Run the 20-Question Benchmark

```bash
python main.py --eval
```

### 4. Run the Interactive Multi-Mode CLI

```bash
export GEMINI_API_KEY="your-gemini-api-key"
python main.py --cli
```

In the CLI, switch modes dynamically using `/mode simple_dense`,
`/mode bm25_sparse`, `/mode hybrid_rrf`, or `/mode hybrid_rerank`.

## Author

**Fatima Azeem** — AI/ML Internship (Phase 3: Production RAG Systems, Day 19)
