# Day 16: Document Ingestion & Chunking Strategies

✅ **2/2 test suite passed** — 100% metadata lineage integrity verified across all 5 chunking strategies.

## Project Overview

A complete multi-format document ingestion and chunking testbed built for enterprise-grade Retrieval-Augmented Generation (RAG). The pipeline processes heterogeneous unstructured documents (Native Vector PDFs, Scanned Image PDFs via OCR, Structured DOCX, and TXT policies), extracts multimodal diagrams, preserves mathematical formulas, and benchmarks five distinct chunking architectures without losing structural metadata lineage.

## Key Features & Architecture

* **Heterogeneous Ingestion Engine**:
  * **Native PDF (`SupportcoursesM-DLearning.pdf`)**: Extracted 117 pages (153,975 chars) using `pdfplumber` and `pymupdf`, capturing 52 distinct hierarchical sections.
  * **Scanned OCR PDF (`vendor_nda_scanned.pdf`)**: 300 DPI pixmap conversion through `pytesseract` + `pymupdf`.
  * **Structured DOCX (`product_spec.docx`)**: Parsed style hierarchy and embedded multi-column SLA tables into Markdown.
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
day-16-chunking-strategies/
├── data/
│   ├── SupportcoursesM-DLearning.pdf
│   ├── vendor_nda_scanned.pdf
│   ├── product_spec.docx
│   └── api_rate_limiting_policy.txt
├── docs/
│   └── TRADEOFF_ANALYSIS.md
├── outputs/
│   ├── images/                           # 96 extracted raster diagrams
│   ├── chunks_fixed_size.json
│   ├── chunks_token_based.json
│   ├── chunks_recursive_langchain.json
│   ├── chunks_semantic.json
│   └── chunks_hierarchical.json
├── src/
│   ├── init.py
│   ├── models.py                         # DocumentElement & TextChunk dataclasses
│   ├── ingestion.py                      # Multi-format parsers & OCR
│   ├── chunkers.py                       # 5 chunking algorithms
│   └── llm_processor.py                  # Table & OCR correction schema
├── tests/
│   └── test_pipeline.py                  # Full metadata validation suite
├── main.py                               # Pipeline execution harness
└── requirements.txt


## Results & Verification

* **2/2 Test Suite Passed**: `pytest tests/` confirms zero missing metadata fields across all generated output JSON files.
* **Extraction Volume**: 162,667 characters total ingested across 4 document types.
* **Lineage Verification**: `PASSED (100% Lineage)` across all 3,429 generated chunks.

## How to Run

### Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
### Run Full Ingestion & Chunking Pipeline
```bash
python main.py

Run Test Suite
Bash

pytest tests/

Strategy Evaluation Verdict

    Hierarchical (Parent-Child) proved to be the highest quality strategy for technical specifications, allowing high-precision vector matches on numerical constraints while feeding full section contexts to the generation model.

    Recursive Character Chunking is the recommended runner-up for single-vector pipelines, preserving Markdown tables and multi-line equations intact.

Author

Fatima Azeem — AI/ML Internship (Phase 3, Day 16)
