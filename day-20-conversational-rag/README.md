# Conversational RAG with Contextual Memory & Query Reformulation — Day 20

## Project Overview

This project implements a production-grade Conversational
Retrieval-Augmented Generation (RAG) system capable of maintaining
multi-turn dialogue context. Standard RAG architectures fail in
conversational settings because follow-up queries frequently rely on
pronouns and elliptical references (e.g., "What is its formula?",
"Compare it with the second one"). This system integrates session-based
conversational memory, LLM-powered query reformulation (anaphora
resolution), persistent vector storage using ChromaDB, and a FastAPI
backend with automated benchmark evaluation.

## Objectives

- Build a multi-turn conversational RAG pipeline that preserves dialogue
  context across multiple interactions.
- Implement query condensation/reformulation to convert
  context-dependent queries into standalone technical search terms.
- Provide isolated, session-based memory management allowing independent
  user dialogue threads.
- Implement strict citation attribution (`[Source: <filename>, Page:
  <page>]`) and negative rejection guardrails.
- Benchmark conversational retrieval performance against raw query
  baselines across multi-turn datasets.
- Expose conversational RAG capabilities via a structured FastAPI
  application and validate functionality with pytest test suites.

## Technologies Used

- Python 3.10+
- LangChain / LangChain Core (conversational chains, message history
  abstractions)
- Google Gemini API (`google-genai` SDK for query condensation and
  response synthesis)
- ChromaDB (persistent vector store and metadata filtering)
- HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- FastAPI & Uvicorn (REST API endpoints for chat orchestration and
  session resets)
- Pytest (automated unit and integration testing)

## Project Structure

```text
day-20-conversational-rag/
├── data/
│   └── multi_turn_conversations.json         # Multi-turn evaluation dataset for benchmarking
├── docs/
│   └── SESSION_MEMORY_ANALYSIS.md             # Technical analysis of session memory trade-offs
├── outputs/
│   ├── chroma_db/                             # Persisted ChromaDB vector database files
│   └── conversational_benchmark_results.json  # Quantitative evaluation metrics & benchmark logs
├── src/
│   ├── __init__.py
│   ├── api.py                                 # FastAPI REST API endpoints (/chat, /reset, /history)
│   ├── conversational_rag.py                  # Core ConversationalRAG pipeline & memory orchestration
│   └── retriever.py                           # ChromaDB vectorstore retrieval & similarity search setup
├── tests/
│   ├── __init__.py
│   └── test_conversational_rag.py             # Pytest suite for memory, reformulation, and API routes
├── main.py                                    # CLI entry point for testing and evaluation workflows
├── README.md
└── requirements.txt
```

## Tasks Performed

### 1. Vector Store & Retrieval Pipeline Setup (`src/retriever.py`)

- Initialized ChromaDB with persistent local storage under
  `outputs/chroma_db/`.
- Indexed technical documentation chunks with structured metadata
  containing source document names, page numbers, and unique chunk IDs.
- Configured similarity search with top-*k* document selection.

### 2. Conversational Engine & Query Reformulation (`src/conversational_rag.py`)

- Built an in-memory session manager tracking dialogue turns per
  `session_id`.
- Developed a prompt-engineered query condenser that converts
  conversational follow-up questions containing pronouns into
  standalone search queries before hitting the vector index.
- Constructed context-grounded response prompts enforcing strict inline
  citations and refusal guardrails when context is insufficient.

### 3. REST API Interface (`src/api.py`)

- Implemented `POST /api/chat` to accept `session_id` and user
  messages, returning the answer, reformulated query, and citations.
- Implemented `POST /api/session/reset` to clear historical context for
  a specific session.
- Implemented `GET /api/session/{session_id}/history` to inspect active
  session turns.

### 4. Benchmarking & Evaluation (`main.py`, `data/multi_turn_conversations.json`)

- Designed a multi-turn conversational benchmark covering single-turn
  lookups, multi-turn pronoun resolutions, topic shifts, and
  out-of-corpus queries.
- Measured retrieval precision and reformulation accuracy, logging
  outputs to `outputs/conversational_benchmark_results.json`.

### 5. Automated Testing (`tests/test_conversational_rag.py`)

- Wrote test suites validating session state isolation, standalone
  query reformulation, and ground truth citation matching.

## Results

Benchmark evaluation summary
(`outputs/conversational_benchmark_results.json`):

- **Anaphora resolution success rate:** ~94.2% across multi-turn
  conversational evaluation pairs.
- **Retrieval hit rate (Top-4):** improved by 41.8% on Turn 2+ follow-up
  queries compared to non-reformulated raw query baselines.
- **Citation precision:** 100% of generated responses included correct
  `[Source: <filename>, Page: <page>]` metadata tags.
- **Hallucination mitigation:** successfully triggered negative
  guardrail responses ("The provided documentation does not contain
  information to answer this question.") on 100% of out-of-corpus test
  cases.

## Observations

- Passing raw conversational follow-ups (e.g., "What are its main
  limitations?") directly to vector embeddings severely degrades vector
  retrieval accuracy because conversational pronouns lack semantic
  density.
- Separating the query reformulation step from the answer synthesis
  step provides full visibility into why specific chunks were
  retrieved, simplifying system debugging.
- Limiting conversational history to a sliding window of the last 4–6
  turns provides sufficient context for pronoun resolution while
  preventing context overflow and minimizing LLM latency.

## Challenges Encountered

- **Over-reformulation on topic shifts** — when users abruptly shifted
  topics without referencing prior context, the condenser occasionally
  hallucinated relationships to earlier topics. Resolved by refining the
  reformulation prompt with explicit rules to keep queries unchanged if
  no conversational dependencies exist.
- **Redundant formula listings** — chunks containing multiple formulas
  caused the LLM to output neighboring formulas. Solved by adding strict
  scope constraint instructions to the synthesis prompt.
- **Rate limits on multi-step calls** — because each conversational turn
  requires two LLM invocations (one for reformulation, one for
  generation), API quota consumption doubled. A check was added to
  bypass the reformulation LLM call entirely on Turn 1 (empty history).

## How to Run

### 1. Environment Setup

```bash
# Clone the repository and enter the directory
cd day-20-conversational-rag

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 3. Run Pipeline Benchmarks & CLI

```bash
python main.py
```

### 4. Run the FastAPI Server

```bash
uvicorn src.api:app --reload --port 8000
```

### 5. Run Unit & Integration Tests

```bash
pytest tests/
```

## Learning Outcomes

- Mastered multi-turn conversational RAG architectures and dynamic
  session state management.
- Gained deep understanding of query condensation techniques for
  resolving conversational ambiguity and anaphora.
- Learned how to decouple conversational context tracking from vector
  similarity retrieval to maximize embedding search recall.
- Acquired practical experience designing comprehensive benchmark
  evaluation datasets for conversational AI pipelines.

## Author

**Fatima Azeem** — AI/ML Internship (Phase 3, Day 20)
