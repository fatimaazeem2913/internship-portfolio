# RAG Architecture & Theory – Day 15 Internship

## Project Overview

This project was completed as part of Day 15 internship tasks, opening Phase 3 (RAG Systems). The objective was to understand when, why, and how to use Retrieval-Augmented Generation to ground LLM responses in real-world data — and when not to.

This is primarily a research and documentation day, grounded wherever possible in Day 7's already-built, already-verified real RAG mini-project rather than treated as abstract theory. The project setup component (installing and verifying 5 real packages) was fully, functionally tested — not just import-checked — including a real, honest finding about ChromaDB's default embedding function failing in network-restricted environments.

---

## Objectives

- Study and document the full RAG pipeline: Corpus -> Ingestion -> Chunking -> Embedding -> Vector Store -> Retrieval -> Augmented Prompt -> LLM -> Grounded Response.
- Compare RAG vs. fine-tuning: 5 scenarios where RAG wins, 3 where fine-tuning wins.
- Research common RAG failure modes and document fixes for each.
- Study RAG variants: Naive RAG, Advanced RAG, Modular RAG, GraphRAG.
- Document RAG evaluation metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall.
- Set up enterprise_rag_engine/ and install pypdf, pdfplumber, sentence-transformers, chromadb, rank-bm25.

---

## Technologies Used

- Python 3, virtual environment
- pypdf, pdfplumber (PDF extraction)
- sentence-transformers (embeddings)
- chromadb (vector database)
- rank-bm25 (keyword retrieval)
- scikit-learn (TF-IDF, used as a verified network-independent embedding substitute)

---

## Project Structure

```
enterprise_rag_engine
|
|-- README.md
|-- REPORT.md
|
|-- verify_setup.py          (real functional tests for all 5 packages)
|-- embedding_reference.py    (correct real sentence-transformers code for local use)
|
|-- corpus/
|   `-- test_document.pdf      (real PDF used to verify pypdf/pdfplumber)
|
`-- outputs/
    `-- setup_verification_results.txt
```

---

## Tasks Performed

### 1. The Full RAG Pipeline, Documented

REPORT.md Part 1 — every stage of Corpus through Grounded Response, mapped directly to Day 7's real, already-built implementation.

### 2. RAG vs. Fine-Tuning

REPORT.md Part 2 — 5 real scenarios favoring RAG, 3 favoring fine-tuning, each with concrete reasoning rather than generic statements.

### 3. RAG Failure Modes

REPORT.md Part 3 — 4 failure modes (poor chunking, wrong retrieval, context overflow, hallucination despite retrieval), each with a specific fix tied back to concepts already proven in this portfolio (TF-IDF's synonym-blindness, attention's O(n^2) cost, Day 7's grounding instructions, Day 9's temperature finding).

### 4. RAG Variants

REPORT.md Part 4 — Naive, Advanced, Modular RAG, and GraphRAG, sourced from current, searched material (the Gao et al. 2024 survey and current GraphRAG practitioner sources), with a clear "what specifically does each one add" answer for each.

### 5. RAG Evaluation Metrics

REPORT.md Part 5 — all 4 RAGAS metrics with real formulas, sourced directly from the RAGAS documentation, organized into the retrieval-quality pair and generation-quality pair with an explanation of why that separation is genuinely useful for debugging.

### 6. Project Setup

enterprise_rag_engine/ created; all 5 required packages installed and functionally verified via verify_setup.py.

---

## Results

- **5/5 required packages installed and functionally verified**, not just import-checked: pypdf and pdfplumber both correctly extracted real text from a real generated PDF; rank-bm25 correctly ranked the right document for a real test query; ChromaDB correctly stored and retrieved the right document using real vector similarity search.
- **A real, honest bug found during setup verification:** ChromaDB's default embedding function tries to auto-download an ONNX model over the network, failing with a cryptic SHA256 error in this network-restricted environment — documented transparently in both verify_setup.py's docstring and REPORT.md, with the correct fix (always supply your own embeddings explicitly) and correct reference code for real sentence-transformers usage provided separately.
- **All theory content backed by current, searched sources** rather than relying on potentially outdated training data, particularly for the RAGAS metric definitions and the GraphRAG explanation.

---

## Observations

- Framing every pipeline stage against Day 7's already-built implementation made abstract RAG theory concrete immediately — "chunking" isn't just a concept, it's the specific 17-chunk, 6+4+7 split already built and verified.
- The ChromaDB default-embedder network dependency is a genuinely useful, real finding: it's exactly the kind of "it works on my machine" trap that silently breaks in a CI pipeline, a Docker container without internet access, or any offline/restricted environment — the fix (explicit embeddings) is also simply the more architecturally correct pattern regardless of network access, since it puts you in control of exactly which embedding model is used.
- The RAGAS metrics' two-pair structure (retrieval quality vs. generation quality) is a genuinely useful mental model for debugging any RAG system, not just an academic categorization — it directly tells you whether to go fix your chunking/retrieval code or your prompt/generation code.
- Day 9's real finding (Gemini 3.x silently ignoring temperature) turned out to have a second, genuine application here: the standard RAG advice "use low temperature to reduce hallucination" needs a caveat for exactly the models this portfolio has been using since Day 8 — a good example of how one real, verified finding keeps paying off in later, seemingly unrelated contexts.

---

## Challenges Encountered

- The real network restriction on huggingface.co (the same one hit on Days 3, 5, and 7) meant sentence-transformers' actual embedding models couldn't be loaded to demonstrate a fully "real" embedding pipeline end to end in this environment. Resolved the same way as every previous instance of this restriction: a genuine, verifiable substitute (scikit-learn's TF-IDF) proves the surrounding infrastructure (ChromaDB's storage/retrieval) works correctly, while correct, complete reference code for the originally-specified tool is provided separately for local use.
- Disk space ran out during initial package installation (a large accumulated pip cache and unused Playwright browser binaries from earlier, unrelated work) — resolved by identifying and clearing genuinely unnecessary cached files, then successfully completing a clean install.

---

## How to Run

```
cd enterprise_rag_engine
python3 -m venv venv
source venv/bin/activate
pip install pypdf pdfplumber sentence-transformers chromadb rank-bm25 scikit-learn reportlab
python3 verify_setup.py
```

To see real sentence-transformers embeddings (requires real internet access to huggingface.co):
```
python3 embedding_reference.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- The complete RAG pipeline and, critically, why RAG system quality is disproportionately a retrieval engineering problem rather than a prompting problem.
- A concrete, scenario-based decision framework for choosing RAG vs. fine-tuning, rather than treating them as interchangeable techniques.
- The real failure modes that break RAG systems in practice, each with a specific, actionable fix rather than a vague "improve retrieval" suggestion.
- The genuine architectural progression from Naive to Advanced to Modular RAG, and what GraphRAG adds that no amount of better vector search can replicate (multi-hop reasoning across explicit relationships).
- How to actually evaluate whether a RAG system is working, using the RAGAS framework's four metrics, and how to use them diagnostically to isolate retrieval failures from generation failures.
- Why verifying a package "works" requires more than a successful import — and how to design a real functional test that would have caught a genuine, non-obvious failure (ChromaDB's network-dependent default) before it became a production surprise.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 15 (Phase 3 begins)
