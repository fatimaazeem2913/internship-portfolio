# enterprise_rag_engine — Day 15: RAG Architecture & Theory

A real, runnable Retrieval-Augmented Generation engine built to accompany
the Day 15 theory deliverables in `docs/`. Every stage of the RAG pipeline
documented in `docs/RAG_STUDY.md` has a corresponding, tested Python module
here — this isn't a diagram-only exercise.

## Structure

```
enterprise_rag_engine/
├── data/
│   ├── corpus/                  # source documents (1 .txt, 1 .pdf)
│   └── vector_store/            # persisted ChromaDB index
├── docs/
│   ├── RAG_STUDY.md             # full pipeline write-up
│   ├── rag_vs_finetuning.md     # 5 scenarios for RAG, 3 for fine-tuning
│   ├── rag_failure_modes.md     # 4 failure modes + fixes, reproduced live
│   ├── rag_variants.md          # Naive / Advanced / Modular / GraphRAG
│   └── rag_evaluation.md        # RAGAS: Faithfulness, Relevance, Precision, Recall
├── src/
│   ├── ingestion.py             # Corpus -> raw text (.txt + .pdf)
│   ├── chunking.py              # Raw text -> paragraph-aware chunks
│   ├── embedding.py             # Chunks -> vectors (sentence-transformers, TF-IDF fallback)
│   ├── vector_store.py          # Vectors -> persisted ChromaDB index
│   ├── retrieval.py             # Query -> ranked chunks (dense / BM25 / hybrid RRF)
│   ├── llm.py                   # Augmented prompt -> grounded answer (Gemini / mock)
│   └── pipeline.py              # Wires every stage together end-to-end
├── tests/
│   └── test_pipeline.py         # 22 tests covering every module
├── venv/                        # dedicated virtual environment for this project
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Running the pipeline

```bash
# Offline / free mode (default) -- no API key or network needed:
USE_MOCK_LLM=true venv/bin/python src/pipeline.py

# Real Gemini mode:
export GEMINI_API_KEY=your_key_here   # free at aistudio.google.com/apikey
venv/bin/python src/pipeline.py
```

## Running the tests

```bash
venv/bin/python -m pytest tests/ -v
```

22/22 tests passing, offline (mock LLM mode, no API key needed to run the
suite).

## Real findings from building this (see REPORT.md for full detail)

- **huggingface.co is not reachable from this sandbox's network allowlist**,
  so `sentence-transformers` can't download its model weights here. A
  network-free TF-IDF fallback (scikit-learn) was built so the pipeline
  still runs end-to-end; the real sentence-transformers code path is left
  in as the correct primary implementation for any environment with normal
  internet access. See `src/embedding.py`'s docstring for the full story.
- **ChromaDB's default embedding function auto-downloads an ONNX model**,
  hitting the same class of network issue. Fixed by always supplying
  embeddings explicitly (`embedding_function=None` on the collection).
- **A real, reproduced example of the "wrong retrieval" failure mode**:
  querying "unopened product" against a corpus that says "unused condition"
  fails under the TF-IDF fallback (zero shared vocabulary) but would
  succeed under real semantic embeddings or the hybrid BM25+dense retriever
  built here. Documented in `docs/rag_failure_modes.md`.
