# Day 16 — Document Ingestion & Chunking Strategies

## Overview

This project builds real ingestion pipelines for 4 document forms (native
PDF, DOCX, TXT, and scanned/image-only PDF via OCR), implements and
compares 5 distinct chunking strategies, and verifies that metadata
(source filename, page number, chunk index, section heading) survives
intact through every stage for every document type and strategy.

## Structure

```
day16_chunking/
├── data/
│   ├── pdfs/employee_handbook.pdf       # real, 56-page generated PDF (55 chapters)
│   ├── docx/product_spec.docx           # real DOCX with headings + table
│   ├── txt/api_rate_limiting_policy.txt # real plain TXT
│   ├── scanned/vendor_nda_scanned.pdf   # real image-only scanned PDF (0 native chars)
│   └── outputs/strategy_comparison_report.json
├── docs/
│   ├── chunking_tradeoffs.md   # required trade-off analysis, real measured data
│   └── ocr_strategy.md         # required OCR/LLM-pass strategy write-up
├── src/
│   ├── ingest_pdf.py            # pdfplumber + PyPDF2, with fallback
│   ├── ingest_docx.py           # python-docx, heading + table aware
│   ├── ingest_txt.py            # ALL-CAPS heading heuristic
│   ├── ingest_ocr.py            # pytesseract, real native-vs-OCR comparison
│   ├── llm_ocr_correction.py    # required LLM pass over raw OCR output
│   ├── chunking_strategies.py   # all 5 strategies
│   ├── embedding.py             # real model + honest fallback (for semantic chunking)
│   ├── pipeline.py              # ties ingestion + chunking + metadata together
│   └── compare_strategies.py    # runs all 5 x 3 doc types, produces real stats
├── tests/
│   └── test_day16.py            # 34 real tests
└── requirements.txt
```

## How to Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# System dependencies (already present in most Linux distros):
# sudo apt install tesseract-ocr poppler-utils

# Generate the real test documents (only needed once)
python gen_pdf.py
python gen_docx.py
python gen_scanned_pdf.py

# Run the test suite
python -m pytest tests/ -v

# Run the full strategy comparison across all 3 document types
python src/compare_strategies.py

# Test OCR vs native extraction directly
python src/ingest_ocr.py

# Test the required LLM correction pass (mock mode by default)
python src/llm_ocr_correction.py

# Real Gemini mode for the LLM correction pass
export GEMINI_API_KEY=your_key_here
USE_MOCK_LLM=false python src/llm_ocr_correction.py
```

## Key Results

- **56-page real PDF**, 55 distinct realistic chapters + a real table + FAQ section
- **34/34 tests passing**
- **Native extraction on the scanned PDF: 0 characters.** OCR: **1,833
  characters recovered.** Real, proven comparison, not simulated.
- **Zero metadata integrity issues** across all 5 strategies × 3 document
  types (20 combinations total)
- Two real, honestly-documented sandbox network limitations (tiktoken's
  encoding file, sentence-transformers' model weights) — same pattern
  established in Day 15, with real fallbacks and clear logging, never
  silent substitution

See `REPORT.md` for the full write-up, and `docs/chunking_tradeoffs.md` /
`docs/ocr_strategy.md` for the required analysis documents.
