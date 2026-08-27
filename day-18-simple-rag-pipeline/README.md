# Simple (Naive) RAG Pipeline & Failure Mode Analysis — Day 18

## Project Overview

This module implements an end-to-end baseline (Naive) Retrieval-
Augmented Generation (RAG) system. The pipeline integrates dense vector
retrieval via **ChromaDB** with LLM generation (`gemini-2.5-flash`),
structured prompt templates, and citation metadata tracking. In addition
to core pipeline execution, this task establishes a comprehensive
evaluation framework to identify, document, and quantify the systemic
failure modes of Naive RAG architectures (such as vocabulary mismatch,
out-of-domain hallucinations, and lost-in-the-middle context
fragmentation).

## Objectives

- **End-to-end pipeline construction** — build a functional RAG pipeline
  integrating document indexing, dense semantic retrieval, dynamic
  prompt synthesis, and LLM generation with exact source attribution.
- **Prompt engineering & specification** — develop structured system
  instructions in `PROMPT_SPECIFICATION.md` enforcing strict context
  grounding, preventing external hallucinations, and requiring inline
  citations.
- **Empirical failure analysis** — identify and catalog critical failure
  modes of baseline RAG in `FAILURE_ANALYSIS.md` (e.g., dense vector
  blindspots, semantic drift, unanswerable queries).
- **Automated evaluation** — benchmark the baseline pipeline across a
  curated query dataset (`eval_questions.json`) and output structured
  results to `outputs/eval_benchmark_results.json`.

## Technologies Used

- **Language & Runtime:** Python 3.12
- **LLM & Inference API:** Google Gemini Flash API (`google-genai` /
  `gemini-2.5-flash`)
- **Vector Store & Embeddings:** ChromaDB (`chromadb`),
  SentenceTransformers (`all-MiniLM-L6-v2` / `bge-large-en-v1.5`)
- **Testing & Evaluation:** Pytest, Rich, Pydantic

## Project Structure

```text
day-18-simple-rag-pipeline/
├── data/
│   ├── chunks_hierarchical.json
│   ├── eval_questions.json
│   └── sample_chunks.json
├── docs/
│   ├── FAILURE_ANALYSIS.md
│   └── PROMPT_SPECIFICATION.md
├── outputs/
│   ├── chroma_db/
│   │   └── chroma.sqlite3
│   └── eval_benchmark_results.json
├── src/
│   ├── __init__.py
│   ├── llm_client.py
│   ├── pipeline.py
│   ├── prompt_builder.py
│   └── retriever.py
├── tests/
│   ├── __init__.py
│   └── test_rag_pipeline.py
├── main.py
├── requirements.txt
└── README.md
```

## Tasks Performed

- **Dense semantic retriever** (`src/retriever.py`) — connected the
  ChromaDB persistent vector store to index hierarchical multimodal
  chunks and execute cosine similarity top-*k* nearest neighbor lookups.
- **Context assembly & prompt builder** (`src/prompt_builder.py`) —
  structured user queries and retrieved context passages into rigorous
  instruction-following prompt templates with source metadata tags
  (`[Source: doc, Page: N]`).
- **Generation & LLM interface** (`src/llm_client.py`) — built an
  abstraction over the Gemini Flash API supporting deterministic
  decoding (temperature = 0.0), token limits, and strict schema
  responses.
- **Orchestrator pipeline** (`src/pipeline.py`) — glued the ingestion,
  retrieval, context injection, and generation lifecycle into a unified
  callable interface (`SimpleRAGPipeline`).
- **Failure mode taxonomy** (`docs/FAILURE_ANALYSIS.md`) — conducted
  failure mode analysis classifying pipeline weaknesses across
  Retrieval, Context Assembly, and Synthesis.
- **Automated test suite** (`tests/test_rag_pipeline.py`) — verified
  embedding ingestion, vector retrieval accuracy, prompt construction,
  and full generative RAG execution.

## Simple RAG Architecture & Workflow

```
[User Query]
     │
     ▼
[Embedding Model] ──(Dense Vector)──► [ChromaDB Index]
                                            │
                                            ▼ (Top-K Chunks)
[Prompt Builder] ◄── [Retrieved Chunks + System Grounding Instructions]
     │
     ▼
[LLM Client (Gemini)] ──► [Grounded Answer + Page-Level Citations]
```

## Systematic Failure Mode Analysis

As documented in `docs/FAILURE_ANALYSIS.md`, baseline (Naive) RAG
exhibits predictable architectural limitations:

1. **Vocabulary mismatch (dense vector blindspots)** — dense bi-encoders
   optimize for semantic concepts but struggle with exact alphanumeric
   codes, acronyms, or specific error numbers (e.g., matching
   `ERROR 403 CSRF` vs. `forbidden request`).
2. **False-positive retrieval (unanswerable queries)** — standard
   top-*k* search unconditionally returns the closest vectors even if
   the cosine similarity is low, forcing irrelevant context into the LLM
   prompt.
3. **Context fragmentation ("lost in the middle")** — splitting long
   passages can sever mathematical formulas from their variable
   definitions, or place premises and conclusions into disparate
   chunks.
4. **Hallucination via extrapolation** — when retrieved context is
   incomplete, LLMs tend to fill knowledge gaps with parametric
   pre-training memory rather than stating that information is missing.

## Results & Benchmark Outputs

- **Dataset evaluated:** 20 specialized technical validation questions
  spanning definitions, LaTeX formulas, diagram anatomy, and adversarial
  unanswerable prompts (`eval_questions.json`).
- **Outputs generated:** successfully benchmarked answers, retrieval
  latency, and context relevance scores saved to
  `outputs/eval_benchmark_results.json`.
- **Test suite:** 100% test pass rate across unit and integration
  assertions in `tests/test_rag_pipeline.py`.

## How to Run

```bash
# 1. Activate virtual environment
source ../day-19-hybrid-search-advanced-retrieval/venv/bin/activate

# 2. Run main RAG execution & evaluation pipeline
python main.py

# 3. Run test suite
python -m pytest tests/
```

## Learning Outcomes

- Built and deployed a complete, modular Naive RAG pipeline using
  ChromaDB and Gemini Flash.
- Developed systematic prompt scaffolding for grounded generation and
  strict citation compliance.
- Diagnosed empirical failure modes of dense-only retrieval,
  establishing the engineering rationale for Day 19: Hybrid Search (BM25
  + Dense RRF) and Day 20: RAG Triad Evaluation Metrics.

## Author

**Fatima Azeem** — AI/ML Internship (Phase 3, Day 18)
