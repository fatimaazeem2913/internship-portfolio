# Day 16: Document Ingestion & Chunking Strategies

✅ **2/2 tests passing** — 100% metadata lineage integrity verified across all 5 chunking strategies.

## Project Overview

A complete multi-format document ingestion and chunking testbed built for enterprise-grade Retrieval-Augmented Generation (RAG). The pipeline processes heterogeneous unstructured documents (Native Vector PDFs, Scanned Image PDFs via OCR, Structured DOCX, and TXT policies), extracts multimodal diagrams, preserves mathematical formulas, and benchmarks five distinct chunking architectures without losing structural metadata lineage.

## Key Features & Architecture

* **Heterogeneous Ingestion Engine**:
  * **Native PDF (`SupportcoursesM-DLearning.pdf`)**: Extracted 117 pages (153,975 chars) using `pdfplumber` and `pymupdf`, capturing 52 distinct hierarchical sections.
  * **Scanned OCR PDF (`vendor_nda_scanned.pdf`)**: 300 DPI pixmap conversion through `pytesseract` + `pymupdf`.
  * **Structured DOCX (`product_spec.docx`)**: Parsed style hierarchy and embedded multi-column SLA tables into clean Markdown.
  * **Policy TXT (`api_rate_limiting_policy.txt`)**: Uppercase header token tracking.
* **Multimodal Extraction**: Extracted and cataloged 96 high-resolution diagrams, architecture topologies, and decision boundaries into `outputs/images/`.
* **Formula & Equation Ingestion**: Preserved spatial mathematical derivations (MSE loss, Logistic probability, Bayes theorem, and Euclidean distance) directly within text chunks.
* **5 Chunking Strategies Benchmarked**:
  1. **Fixed-Size Chunking**: 451 chunks (500 chars / 50 overlap).
  2. **Token-Based Chunking**: 402 chunks via `tiktoken` BPE (`cl100k_base`).
  3. **Recursive Character (LangChain)**: 431 chunks using hierarchical delimiters (`\n\n`, `\n`, `. `).
  4. **Semantic Chunking**: 1,046 sentence-boundary clusters.
  5. **Hierarchical (Parent-Child)**: 1,099 small child chunks (200 chars) mapped to rich parent contexts (800 chars).
* **100% Lineage Integrity**: Every chunk maintains strict provenance (`source`, `page_number`, `chunk_index`, and `section_heading`).

## Project Structure

```text
day_16_rag_ingestion_chunking/
├── data/                                 # Source input files (PDF, DOCX, TXT, OCR)
│   ├── SupportcoursesM-DLearning.pdf
│   ├── vendor_nda_scanned.pdf
│   ├── product_spec.docx
│   └── api_rate_limiting_policy.txt
├── docs/                                 # Technical reports & Trade-off analyses
│   └── TRADEOFF_ANALYSIS.md
├── outputs/                              # Exported chunk JSON files & figures
│   ├── images/                           # 96 extracted raster diagrams
│   ├── chunks_fixed_size.json
│   ├── chunks_token_based.json
│   ├── chunks_recursive_langchain.json
│   ├── chunks_semantic.json
│   └── chunks_hierarchical.json
├── src/                                  # Core pipeline modules
│   ├── __init__.py
│   ├── models.py                         # DocumentElement & TextChunk dataclasses
│   ├── ingestion.py                      # Multi-format parsers & OCR
│   ├── chunkers.py                       # 5 chunking algorithms
│   └── llm_processor.py                  # Table & OCR correction schema
├── tests/                                # Verification and metadata integrity tests
│   ├── __init__.py
│   └── test_pipeline.py
├── main.py                               # Master execution runner
└── requirements.txt                      # Python dependencies
