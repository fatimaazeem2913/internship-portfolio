# Day 16: Advanced Document Ingestion & Chunking Strategies Pipeline

Production-grade RAG ingestion, OCR processing, LLM post-processing, and multi-strategy chunking testbed.

## Overview
This repository implements a modular ingestion and chunking architecture for multi-modal, heterogeneous documents:
- **PDF Extraction**: Native multi-page vector PDF extraction (PDFPlumber + PyPDF) with table layout parsing.
- **OCR Extraction**: Scanned PDF & image text parsing with Tesseract/PaddleOCR support.
- **DOCX & TXT Ingestion**: Structural heading hierarchy detection and clean text aggregation.
- **LLM-assisted Post-Processing**: Markdown table recovery, JSON generation for lookups, and executive table summaries.
- **5 Chunking Strategies**:
  1. Fixed-size Character Chunking (sliding window with overlap)
  2. Token-based Chunking (tiktoken `cl100k_base`)
  3. Recursive Character Chunking (hierarchical delimiter splitting)
  4. Semantic Chunking (sentence boundary & semantic shift grouping)
  5. Hierarchical / Parent-Child Chunking (parent context indexing for child search units)
- **Rich Metadata Attachment**: Every chunk preserves `source`, `page_number`, `chunk_index`, and `section_heading`.

---

## Directory Structure
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
```

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
```

### Run Master Pipeline
```bash
python main.py
```

### Run Test Suite
```bash
pytest tests/
```

## Strategy Evaluation Verdict

* **Hierarchical (Parent-Child)** proved to be the highest quality strategy for technical specifications, allowing high-precision vector matches on numerical constraints while feeding full section contexts to the generation model.
* **Recursive Character Chunking** is the recommended runner-up for single-vector pipelines, preserving Markdown tables and multi-line equations intact.

## Author

Fatima Azeem — AI/ML Internship (Phase 3, Day 16)

