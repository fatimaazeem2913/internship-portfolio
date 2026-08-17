# RAG Architecture & Theory — Study Notes

## Objective

Understand when, why, and how to use Retrieval-Augmented Generation (RAG) to
ground LLM responses in real-world data — and when *not* to use it.

## What is RAG?

An LLM's knowledge is frozen at training time and lives entirely inside its
weights. RAG fixes two problems that causes:

1. **Staleness** — the model can't know about anything that happened, or
   any document that was written, after training.
2. **Hallucination on unknown facts** — asked about something it never saw,
   an LLM will often generate a fluent, confident, *wrong* answer rather
   than say "I don't know."

RAG solves both by retrieving relevant real documents at query time and
handing them to the LLM as context, so the model is *grounding* its answer
in text it can actually point to, instead of generating purely from its
frozen internal knowledge.

## The Full Pipeline

```
Corpus → Ingestion → Chunking → Embedding → Vector Store → Retrieval
       → Augmented Prompt → LLM → Grounded Response
```

| Stage | What happens | This project's module |
|---|---|---|
| **Corpus** | The raw source-of-truth documents (policy docs, PDFs, wikis, tickets, etc.) | `data/corpus/` |
| **Ingestion** | Read each file, extract plain text regardless of source format | `src/ingestion.py` |
| **Chunking** | Split long documents into smaller, retrievable pieces | `src/chunking.py` |
| **Embedding** | Convert each chunk's text into a dense numeric vector that captures its meaning | `src/embedding.py` |
| **Vector Store** | Persist chunk vectors + text + metadata in a searchable index | `src/vector_store.py` |
| **Retrieval** | At query time, embed the question and find the most relevant chunks | `src/retrieval.py` |
| **Augmented Prompt** | Stitch retrieved chunks + the question into one prompt for the LLM | `src/llm.py::build_augmented_prompt` |
| **LLM** | Generate an answer conditioned on that prompt | `src/llm.py::generate_answer` |
| **Grounded Response** | The final answer, plus which source(s) it came from | pipeline output |

This is the same 9-stage shape used in Day 7's mini-RAG project, now built
as a real, testable, modular engine instead of a single notebook-style script.

### Why chunking exists at all

Embedding models have a limited effective context window, and retrieval
precision drops as chunks get longer (a long chunk mixes multiple topics,
so a query about one narrow fact pulls in a lot of irrelevant surrounding
text). Chunking is the tradeoff knob between **precision** (small chunks,
very targeted, but each chunk may lack surrounding context) and **recall**
(large chunks, more context per hit, but noisier matches).

### Why a vector store, not just a Python list

A vector store like ChromaDB adds an approximate-nearest-neighbor index
(HNSW) on top of raw vectors, so similarity search stays fast even as the
corpus grows to millions of chunks — a linear scan over embeddings does not
scale, but HNSW-indexed lookup does.

### Why retrieval is "search," not "SQL"

Retrieval ranks chunks by *similarity to the query's meaning* (cosine
distance between embeddings), not by exact keyword match — which is exactly
why RAG can answer a question phrased totally differently from the source
document's wording, as long as an embedding model with real semantic
understanding is used (see `rag_failure_modes.md` for what happens when it
isn't).

## Reference implementation

`enterprise_rag_engine/` in this project directory is a real, running
implementation of every stage above, built and tested against a real
2-document corpus (a .txt and a genuinely generated .pdf). See
`enterprise_rag_engine/README.md` for how to run it.
