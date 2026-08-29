# Voice-Enabled Enterprise RAG System with Local Whisper STT — Day 22

## Project Overview

This project delivers a multi-modal, Voice-Enabled Enterprise
Retrieval-Augmented Generation (RAG) system. Building upon
multi-strategy hybrid retrieval, this iteration integrates an offline,
privacy-preserving Speech-to-Text (STT) pipeline using `faster-whisper`
(CTranslate2 with `int8` quantization). The application features a
browser-native voice recording interface directly inside the search
bar, server-side acoustic transcription with Voice Activity Detection
(VAD), query reformulation to correct acoustic ambiguities,
multi-strategy document retrieval (Dense, BM25, Hybrid RRF, Hierarchical
Compression), and LaTeX-rendered grounded responses with active
metadata citations.

## Objectives

- Integrate a high-performance, local Speech-to-Text engine using
  `faster-whisper` without third-party cloud audio dependencies.
- Build an intuitive voice-enabled chat input in the React frontend
  using the browser `MediaRecorder` API.
- Implement server-side audio ingestion and transcription pipelines
  handling binary audio blobs (`audio/webm`) with Voice Activity
  Detection (VAD) filtering.
- Enable voice-to-text pre-population in the input search bar, allowing
  manual user inspection and editing before RAG query submission.
- Maintain conversational anaphora and phonetic query resolution through
  LLM query reformulation.
- Ground voice-derived queries across ingested enterprise documents with
  inline source citations and negative guardrails.

## Technologies Used

- **Speech-to-Text (STT):** `faster-whisper` (CTranslate2, `base` model,
  `int8` CPU quantization, Silero VAD)
- **Backend Framework:** Python 3.10+, FastAPI, Uvicorn, Python
  Multipart
- **LLM & Embeddings:** Google Gemini API (`google-genai`), HuggingFace
  Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Store & Retrieval:** ChromaDB, `rank-bm25` (BM25Okapi),
  LangChain Community / Core
- **Document Processing:** PyPDF2 / pdfplumber, python-docx
- **Frontend Dashboard:** React 18, Vite, Browser MediaStream /
  MediaRecorder Web API, KaTeX (`remark-math`, `rehype-katex`,
  `react-markdown`)
- **Testing & Evaluation:** Pytest

## Project Structure

```text
day-22-voice-rag/
├── backend/
│   ├── data/
│   │   ├── uploads/                     # Storage for ingested documents
│   │   ├── evaluation_set.json          # Benchmark evaluation queries
│   │   └── sample_corpus.json           # Default pre-indexed documentation corpus
│   ├── outputs/
│   │   ├── chroma_db/                   # Persistent ChromaDB vector database
│   │   └── evaluation_matrix.json       # Quantitative retrieval benchmark logs
│   ├── src/
│   │   ├── __init__.py
│   │   ├── api.py                       # FastAPI routes (/api/transcribe, /api/rag/chat, /sources)
│   │   ├── ingestion.py                 # Multi-format document parsing & chunking engine
│   │   ├── rag_service.py               # Conversational query reformulation & LLM generation
│   │   ├── strategies.py                # Dense, BM25, Hybrid RRF, & Hierarchical retrievers
│   │   └── stt_service.py               # Local faster-whisper STT engine & VAD filtering
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_api.py                  # API endpoints and audio transcription tests
│   ├── main.py                          # CLI execution & evaluation harness
│   ├── requirements.txt                 # Backend Python dependencies
│   └── test_llm.py                      # LLM connectivity and quota testing utility
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/                  # UI components and control modules
│   │   ├── App.jsx                      # Voice-enabled chat plane, audio capture, KaTeX rendering
│   │   ├── index.jsx                    # React application entry point
│   │   └── styles.css                   # Enterprise dark-theme interface styling
│   ├── index.html                       # HTML template
│   ├── package-lock.json
│   ├── package.json                     # Frontend dependencies & scripts
│   └── vite.config.js                   # Vite configuration
└── README.md
```

## Tasks Performed

### 1. Local Speech-to-Text Microservice (`backend/src/stt_service.py`)

- Initialized `faster-whisper` using the lightweight `base` model on CPU
  with `int8` quantization to minimize latency and memory overhead.
- Built safe temporary audio buffer handlers to ingest binary chunks
  from browser streams.
- Enabled Silero Voice Activity Detection (`vad_filter=True`) to
  automatically remove ambient background noise and speech pauses.

### 2. Audio Transcription REST Endpoint (`backend/src/api.py`)

- Implemented `POST /api/transcribe` accepting `UploadFile` payloads
  (`audio/webm`).
- Streamed binary audio bytes directly into the STT service and
  returned structured JSON containing the transcribed text string and
  language metadata.

### 3. Unified Voice & Text UI Input Bar (`frontend/src/App.jsx`)

- Built a streamlined chat bar integrating text input, real-time
  recording timer, microphone trigger button, and send button.
- Captured microphone audio using `navigator.mediaDevices.getUserMedia`
  and `MediaRecorder` with `audio/webm;codecs=opus` encoding.
- Populated transcribed voice queries directly into the search bar upon
  recording completion for user verification prior to dispatch.

### 4. Acoustic & Conversational Query Reformulation (`backend/src/rag_service.py`)

- Enhanced the LLM query condenser to resolve both multi-turn
  conversational pronouns ("it", "the second metric") and minor acoustic
  speech recognition transcription errors.
- Enforced strict formula scoping and citation grounding instructions in
  the synthesis prompt.

### 5. Multi-Strategy Corpus Retrieval & Document Management (`backend/src/strategies.py`, `backend/src/ingestion.py`)

- Maintained dynamic ChromaDB indexing and sparse BM25 recalculation
  across uploaded PDF, DOCX, and TXT files.
- Added `get_indexed_sources()` to aggregate chunk counts across
  ingested documents for live sidebar inspection.

## Results

- **STT transcription latency:** average turnaround time of ~0.6–1.1s
  for a 5-second voice query on standard CPU hardware using `int8`
  quantization.
- **Acoustic robustness:** VAD filtering successfully eliminated
  leading/trailing silence, improving transcription quality on
  technical terminology (e.g., Support Vector Machines, Mean Squared
  Error).
- **Retrieval & grounding accuracy:** 100% adherence to inline citations
  (`[Source: <filename>, Page: <page>]`) and mathematical LaTeX
  rendering across verified technical answers.
- **Negative guardrail adherence:** correctly triggered refusal
  fallbacks on out-of-scope audio queries without hallucinating details.

## Observations

- Quantizing the Whisper model weights to `int8` reduces RAM utilization
  by roughly 4x while maintaining high transcription fidelity for
  technical and domain-specific terms.
- Placing the microphone directly into the search bar and populating
  transcribed text for user review before submission significantly
  mitigates accidental query dispatches caused by acoustic
  misinterpretations.
- Query reformulation acts as an effective secondary error-correction
  layer for slight homophone misspellings before embedding generation.

## Challenges Encountered

- **Acoustic homophones on technical acronyms** — fast speech caused
  Whisper to occasionally transcribe SVM or SVD as phonetic variants.
  Resolved by passing conversation context through query reformulation
  to correct technical phrasing before database retrieval.
- **Browser audio codec heterogeneity** — different browsers record
  audio using varying MIME types. Handled by verifying
  `MediaRecorder.isTypeSupported("audio/webm;codecs=opus")` before
  falling back to default audio container formats.
- **API key rate-limiting** — frequent voice testing caused rapid
  free-tier quota exhaustion across multi-step chains. Solved by
  bypassing the query reformulation call on cold-start queries (empty
  session history).

## How to Run

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (including faster-whisper and langchain)
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

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```

Open `http://localhost:5173` in your browser.

### 3. Run Automated Tests

```bash
cd backend
pytest tests/
```

## Learning Outcomes

- Built and deployed an end-to-end voice-enabled conversational RAG
  application.
- Gained hands-on experience running optimized local Speech-to-Text
  inference using `faster-whisper` and CTranslate2 quantization.
- Implemented browser-level audio recording via Web Audio and
  `MediaRecorder` APIs.
- Mastered full-stack state coordination across audio streaming, speech
  transcription, query reformulation, and vector database retrieval.

## Author

**Fatima Azeem** — AI/ML Internship (Phase 3, Day 22)
