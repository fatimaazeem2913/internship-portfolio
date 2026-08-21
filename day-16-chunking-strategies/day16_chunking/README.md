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
```
day_16_rag_ingestion_chunking/
├── data/                       # Source input files (PDF, DOCX, TXT, OCR)
├── docs/                       # Technical reports & Trade-off analyses
│   └── TRADEOFF_ANALYSIS.md
├── outputs/                    # Exported chunk JSON files & benchmark logs
├── src/                        # Core pipeline modules
│   ├── __init__.py
│   ├── models.py               # DocumentElement and Chunk dataclasses
│   ├── ingestion.py            # PDF, OCR, DOCX, and TXT extractors
│   ├── chunkers.py             # 5 Chunking strategies implementations
│   └── llm_processor.py        # Table formatting & LLM correction layer
├── tests/                      # Verification and metadata integrity tests
│   ├── __init__.py
│   └── test_pipeline.py
├── main.py                     # Master execution runner
└── requirements.txt            # Python dependencies
```

---

## Installation & Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline
python main.py

# 3. Run test suite
pytest tests/
```
