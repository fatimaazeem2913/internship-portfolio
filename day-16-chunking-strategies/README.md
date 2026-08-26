# Document Ingestion, Multimodal Parsing & Chunking Strategies — Day 16

## Project Overview

This module establishes the foundational data preparation layer for a production Retrieval-Augmented Generation (RAG) system. It processes heterogeneous enterprise document formats — plain text policies (`.txt`), word processing specifications (`.docx`), scanned legal contracts, and technical course curriculum PDFs containing dense descriptive prose, tabular benchmarks, and visual architecture diagrams — into standardized, searchable chunks while preserving complete metadata lineage and context provenance.

## Objectives

- Ingest multi-format unstructured documents (`.txt`, `.docx`, `.pdf`, scanned PDFs) into unified data representations.
- Implement and benchmark 5 distinct chunking strategies: Fixed-Size, Token-Based, Recursive Character, Semantic Coherence, and Hierarchical Parent-Child chunking.
- Integrate multimodal vision models to transcribe embedded figures, neural diagrams, and loss curves into detailed, semantically searchable markdown descriptions.
- Preserve document metadata (source filenames, page numbers, section headers, chunk IDs, and hierarchical parent-child relationships) to eliminate orphan chunks.

## Technologies Used

- **Languages & Runtime:** Python 3.12, PyPDF2, PyMuPDF (`fitz`), `python-docx`
- **Vision & Tokenization:** Google Gemini Flash Vision API (`gemini-2.5-flash`), `tiktoken`, `langchain-text-splitters`
- **Data Architecture & Scaffolding:** Pydantic models, NumPy, Rich, Tabulate

## Project Structure

```text
day-16-chunking-strategies/
├── data/
│   ├── api_rate_limiting_policy.txt
│   ├── product_spec.docx
│   ├── SupportcoursesM-DLearning.pdf
│   └── vendor_nda_scanned.pdf
├── docs/
│   └── TRADEOFF_ANALYSIS.md
├── outputs/
│   ├── images/
│   ├── chunks_fixed_size.json
│   ├── chunks_hierarchical.json
│   ├── chunks_recursive_langchain.json
│   ├── chunks_semantic.json
│   └── chunks_token_based.json
├── src/
│   ├── __init__.py
│   ├── chunkers.py
│   ├── document_parser.py
│   ├── ingestion.py
│   ├── llm_processor.py
│   └── models.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Tasks Performed

- Universal Document Ingestion: Built universal loaders in ingestion.py and document_parser.py supporting multi-format document streams (.pdf, .docx, .txt).

- Visual Extraction & Vision Enrichment: Extracted and routed embedded images to outputs/images/, producing vision-transcribed textual markdown descriptions via llm_processor.py.

- Modular Chunking Implementations: Implemented modular chunking algorithms in chunkers.py:

    1. Fixed-Size Windowing: Uniform character splitting with overlap.

    2. Token-Based Splitting: BPE token-aware segmenting via tiktoken.

    3. Recursive LangChain Splitting: Structural boundary preservation prioritizing double-newlines, single-newlines, and whitespace breaks.

    4. Semantic Chunking: Similarity-threshold grouping based on embedding distance deltas.

    5. Hierarchical Parent-Child Chunking: Granular child passages (150–300 tokens) linked to broad parent contexts (800–1200 tokens) via strict ID references.

- Schema Validation: Validated chunk schemas with Pydantic (models.py) and saved serialized outputs in outputs/.

## Results

- Generated **280 standardized multimodal chunks** from the target
  curriculum corpus (`SupportcoursesM-DLearning.pdf`).
- Structured **100% of image figures** (e.g., biological neuron
  anatomy, MSE loss landscapes, gradient convergence trajectories) into
  fully indexed Markdown text passages with exact page-level provenance.

## Observations

- Fixed-size windowing consistently splits mathematical expressions
  across chunk boundaries, resulting in broken LaTeX strings.
- Hierarchical parent-child chunking provides the highest semantic
  precision by allowing fine-grained child matching while preserving
  parent narrative context during generation.

## Challenges Encountered

- **Handling inline mathematical formulas** — splitting text mid-equation
  rendered LaTeX blocks unparseable. Resolved by enforcing boundary
  splits at paragraph and double-newline deltas before falling back to
  token limits.
- **Complex multi-column PDF formatting** — reading order distortion was
  mitigated using PyMuPDF block extraction sorted by vertical reading
  coordinates.

## How to Run

```bash
# 1. Activate Virtual Environment
source ../day-19-hybrid-search-advanced-retrieval/venv/bin/activate

# 2. Run Parsing & Chunking Pipeline
python main.py

# 3. Run Test Suite
python -m pytest tests/
```

## Learning Outcomes

- Developed production-grade document extraction pipelines across multi-format enterprise data.

- Evaluated trade-offs among 5 distinct chunking architectures for retrieval-augmented generation.

- Implemented multimodal vision transcription for indexing technical diagrams and figures.

## Author

**Fatima Azeem** — AI/ML Internship (Phase 3, Day 16)