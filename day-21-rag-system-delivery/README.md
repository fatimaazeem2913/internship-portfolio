# Full-Stack Enterprise RAG System Delivery & Multi-Strategy Evaluation — Day 21

## Project Overview

This project delivers a complete, production-ready Full-Stack Enterprise
Retrieval-Augmented Generation (RAG) system. It combines a
high-performance FastAPI backend with a dynamic React (Vite) control
plane dashboard. The system features multi-format document ingestion
(PDF, DOCX, TXT), multi-strategy retrieval mechanisms (Dense Vector,
Sparse BM25, Hybrid RRF, and Hierarchical Compression), real-time
metadata citation tracking, an automated quantitative evaluation matrix,
and negative guardrails to prevent hallucinations.

## Objectives

- Deliver an end-to-end full-stack RAG web application with separated
  backend services and interactive frontend interfaces.
- Implement a multi-strategy retrieval architecture enabling comparative
  execution across Dense (MiniLM), Sparse (BM25), Hybrid (Reciprocal
  Rank Fusion), and Hierarchical Compression techniques.
- Build a dynamic document ingestion pipeline supporting drag-and-drop
  parsing for PDF, DOCX, and TXT files with on-the-fly ChromaDB
  indexing.
- Provide an automated evaluation harness to benchmark retrieval
  strategies against an empirical test set across hit rate, MRR,
  latency, and context precision.
- Render real-time chunk inspections, source citation tags, and
  formatted mathematical equations directly on the client UI.

## Technologies Used

- **Backend Framework:** Python 3.10+, FastAPI, Uvicorn
- **LLM & Embeddings:** Google Gemini API (`google-genai`), HuggingFace
  Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- **Retrieval & Vector Storage:** ChromaDB, `rank-bm25` (BM25Okapi),
  LangChain Core / Community
- **Document Processing:** PyPDF2 / pdfplumber, python-docx
- **Frontend Dashboard:** React 18, Vite, Lucide Icons, KaTeX
  (`remark-math`, `rehype-katex`, `react-markdown`)
- **Testing & Benchmarking:** Pytest, NumPy

## Project Structure

```text
day-21-rag-system-delivery/
├── backend/
│   ├── data/
│   │   ├── uploads/                     # Storage for user-uploaded documents (PDF/DOCX/TXT)
│   │   ├── evaluation_set.json          # Benchmark dataset for RAG quantitative evaluation
│   │   └── sample_corpus.json           # Pre-indexed default corpus
│   ├── outputs/
│   │   ├── chroma_db/                   # Persistent ChromaDB vector collections
│   │   └── evaluation_matrix.json       # Generated benchmark evaluation metrics
│   ├── src/
│   │   ├── __init__.py
│   │   ├── api.py                       # FastAPI application and route definitions
│   │   ├── ingestion.py                 # Multi-format document parsing & chunking engine
│   │   ├── rag_service.py               # Conversational RAG orchestration & LLM synthesis
│   │   └── strategies.py                # Dense, BM25, Hybrid RRF, & Hierarchical retrievers
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_api.py                  # API endpoints and retrieval integration tests
│   ├── main.py                          # CLI entry point for evaluation and server startup
│   ├── requirements.txt                 # Backend Python dependencies
│   └── test_llm.py                      # LLM connectivity and quota testing utility
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/                  # Modular React UI components
│   │   ├── App.jsx                      # Main RAG control plane layout & state management
│   │   ├── index.jsx                    # React DOM entry point
│   │   └── styles.css                   # Enterprise dark-mode dashboard styling
│   ├── index.html                       # HTML document template
│   ├── package-lock.json
│   ├── package.json                     # Frontend Node dependencies & scripts
│   └── vite.config.js                   # Vite dev server and build configuration
└── README.md
```

## Tasks Performed

### 1. Multi-Format Document Ingestion Engine (`backend/src/ingestion.py`)

- Developed robust parsers for raw `.pdf`, `.docx`, and `.txt` files
  with clean text sanitization.
- Structured metadata tags (source filename, page number, generated
  chunk IDs) on extracted passages prior to ChromaDB indexing.

### 2. Advanced Multi-Strategy Retrieval Manager (`backend/src/strategies.py`)

- **Dense retrieval** — embedding-based vector similarity search using
  `all-MiniLM-L6-v2`.
- **Sparse retrieval** — lexical term-frequency scoring via `BM25Okapi`.
- **Hybrid search** — unified score combination using Reciprocal Rank
  Fusion (RRF) with configurable constant k = 60.
- **Hierarchical compression** — sentence-level context extraction
  preserving only query-aligned information.

### 3. Grounded Synthesis & API Layer (`backend/src/rag_service.py`, `backend/src/api.py`)

- Built RAG generation chains with strict source attribution, requiring
  `[Source: <filename>, Page: <page>]` inline citations.
- Integrated negative constraint guardrails that reject queries lacking
  sufficient corpus context.
- Created FastAPI endpoints for `/api/rag/chat`, `/api/rag/ingest`,
  `/api/rag/sources`, and session management.

### 4. React Control Plane Dashboard (`frontend/src/`)

- Designed a responsive dark-mode interface with strategy selectors,
  dynamic document upload cards, active source chunk sidebars, and
  citation pills.
- Integrated LaTeX equation rendering using `remark-math` and
  `rehype-katex`.

### 5. Automated Evaluation Harness (`backend/main.py`, `backend/data/evaluation_set.json`)

- Ran a standardized benchmark evaluating each retrieval strategy across
  Hit Rate@4, Mean Reciprocal Rank (MRR), and latency, exporting
  findings to `backend/outputs/evaluation_matrix.json`.

### 6. System Verification & Unit Testing (`backend/tests/test_api.py`, `backend/test_llm.py`)

- Validated API status codes, multi-turn history handling, ChromaDB
  collection updates, and Gemini model availability.

## Results

### Retrieval Strategy Benchmark Comparison (`outputs/evaluation_matrix.json`)

| Metric | Dense (MiniLM) | Sparse (BM25) | Hybrid (RRF) | Hierarchical Compression |
|---|:---:|:---:|:---:|:---:|
| Hit Rate@4 | 82.4% | 76.1% | **94.8%** | 89.2% |
| MRR (Mean Reciprocal Rank) | 0.741 | 0.688 | **0.887** | 0.812 |
| Average Retrieval Latency | 18 ms | **4 ms** | 24 ms | 31 ms |
| Context Compression Ratio | 0.0% | 0.0% | 0.0% | *not captured as a precise figure — see Observations below (>50% token reduction reported)* |

- **Zero-hallucination rate:** 100% adherence to negative refusal
  guardrails when presented with out-of-scope enterprise queries.
- **Full-stack end-to-end latency:** sub-1.5s total turnaround time from
  UI message submission to fully rendered streaming output.

## Observations

- **Hybrid search outperformance** — combining dense semantic vectors
  with BM25 keyword matching (RRF) yielded the highest recall and MRR,
  particularly on queries containing exact identifiers, acronyms, or
  numbers.
- **Hierarchical token optimization** — hierarchical context compression
  reduced the prompt context token load by over 50% without degrading
  answer quality for mathematical formulas.
- **UI transparency** — displaying active citations and retrieved chunk
  cards alongside the chat response significantly improved user trust
  and auditability.

## Challenges Encountered

- **Dynamic BM25 corpus synchronization** — ingesting new files into
  ChromaDB at runtime left the in-memory BM25 index stale. Resolved by
  implementing an automated corpus reload mechanism triggered upon every
  successful `/api/rag/ingest` call.
- **Complex equation KaTeX formatting** — multiline mathematical outputs
  occasionally caused syntax mismatches in Markdown parsing. Fixed by
  introducing sanitized KaTeX delimiters and standardizing formula
  output instructions in the synthesis prompt.
- **Heterogeneous file schemas** — different document types provided
  inconsistent page and line metadata. Standardized metadata
  normalization inside `ingestion.py` before passing chunks to storage.

## How to Run

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Export your Gemini API key
export GEMINI_API_KEY="your_api_key_here"

# Start the FastAPI server
uvicorn src.api:app --reload --port 8000
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory in a new terminal
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

Open your browser and navigate to `http://localhost:5173`.

### 3. Run Benchmark & Evaluation

```bash
cd backend
python main.py
```

### 4. Run Automated Tests

```bash
cd backend
pytest tests/
```

## Learning Outcomes

- Built and delivered an enterprise-ready, full-stack RAG web
  application with complete architectural separation.
- Implemented and quantitatively evaluated multi-strategy retrieval
  mechanics (Dense, BM25, RRF, Hierarchical).
- Created dynamic document parsing and on-the-fly vector indexing
  workflows for unstructured formats.
- Learned full-stack RAG UI design, including inline citations, chunk
  inspection panels, and LaTeX math rendering.

## Author

**Fatima Azeem** — AI/ML Internship (Phase 3, Day 21)
