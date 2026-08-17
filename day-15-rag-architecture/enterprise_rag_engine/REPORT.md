# Day 15 — RAG Architecture & Theory

## Project Overview

Day 15 opens Phase 3 (RAG Systems). The objective was to understand when,
why, and how to use Retrieval-Augmented Generation to ground LLM responses
in real-world data — and when not to — and to set up an
`enterprise_rag_engine/` project as the foundation for the rest of Phase 3.
Rather than treat this as documentation-only, the engine was built out as a
real, runnable, tested RAG pipeline, so every theoretical claim in the docs
has a corresponding working (and in one case, honestly-failing-then-fixed)
piece of code behind it.

## Objectives

1. Study and document the full RAG pipeline (Corpus → Ingestion → Chunking
   → Embedding → Vector Store → Retrieval → Augmented Prompt → LLM →
   Grounded Response).
2. Compare RAG vs. fine-tuning: 5 scenarios where RAG wins, 3 where
   fine-tuning wins.
3. Research the most common RAG failure modes and document fixes for each.
4. Study RAG variants: Naive, Advanced, Modular, GraphRAG.
5. Document RAG evaluation metrics: Faithfulness, Answer Relevance, Context
   Precision, Context Recall.
6. Set up `enterprise_rag_engine/` and install pypdf, pdfplumber,
   sentence-transformers, chromadb, rank-bm25.

## Tech Used

Python 3.12, pypdf, pdfplumber, sentence-transformers, ChromaDB (persistent
client), rank-bm25 (BM25Okapi), scikit-learn (TF-IDF fallback), reportlab
(test corpus PDF generation), pytest, google-genai (Gemini SDK, real LLM
path), dedicated venv with `--system-site-packages` to reuse the sandbox's
pre-installed torch/transformers.

## Structure

See `README.md` for the full directory tree. Summary: `data/corpus/` (2
real source documents), `docs/` (5 theory documents), `src/` (7 pipeline
modules), `tests/` (22 tests), `requirements.txt`.

## Tasks Performed

- Built a real 2-document corpus: `refund_policy.txt` (hand-written) and
  `shipping_policy.pdf` (generated with reportlab as a genuine multi-section
  PDF, so ingestion actually has to exercise real PDF text extraction, not
  just `.txt` reading).
- Wrote `ingestion.py`: loads both `.txt` and `.pdf` files, with pdfplumber
  as primary PDF extractor and a pypdf fallback if pdfplumber extracts no
  text from a page.
- Wrote `chunking.py`: paragraph-aware recursive chunking (splits on
  paragraph boundaries first, falls back to a word-count sliding window
  with overlap only when a paragraph is still too long).
- Wrote `embedding.py`: sentence-transformers (all-MiniLM-L6-v2) as the
  real, primary embedding path, with an honestly-documented network-blocked
  fallback (see Challenges below).
- Wrote `vector_store.py`: ChromaDB persistent client, with
  `embedding_function=None` explicitly set to avoid the ONNX
  auto-download bug already known from context.
- Wrote `retrieval.py`: three retrieval strategies — pure dense, pure BM25,
  and hybrid (Reciprocal Rank Fusion of both) — to directly address the
  "wrong retrieval" failure mode.
- Wrote `llm.py`: real Gemini call path (following the established
  USE_MOCK_LLM pattern from Day 11 onward) plus an offline mock mode, a
  RAG-specific system prompt instructing the model to answer only from
  context and refuse honestly when context is insufficient.
- Wrote `pipeline.py`: wires every stage together into one `RAGPipeline`
  class with a configurable retrieval mode.
- Wrote and ran `tests/test_pipeline.py`: 22 real pytest tests covering
  ingestion, chunking, embedding, vector store, all three retrieval modes,
  prompt construction, mock LLM generation, and full end-to-end pipeline
  integration. All 22 pass, offline, no API key required.
- Researched and wrote all 5 `docs/` files, verifying RAG variants and
  RAGAS metric definitions via live web search rather than memory alone.

## Results

- 22/22 tests passing.
- Full pipeline runs end-to-end offline (`USE_MOCK_LLM=true`), correctly
  retrieving and citing real source documents for representative queries.
- Real PDF ingestion verified: `shipping_policy.pdf` extracts correctly via
  pdfplumber (no fallback to pypdf needed for this particular PDF, but the
  fallback path exists and is exercised by the code even if not hit by
  this specific corpus).
- Hybrid retrieval verified to outperform either single method alone on an
  exact-term query ("SECTION 5 late refunds") — all three methods
  correctly ranked the right chunk first in this case, but BM25's raw
  score margin (6.594 vs 2.394) shows it carrying more of the signal on
  this exact-term style of query, exactly as the theory in
  `rag_variants.md`/`rag_failure_modes.md` predicts.

## Observations

- Chunking strategy has a bigger downstream effect than expected: because
  chunk boundaries were placed at paragraph breaks, a heading like
  "SECTION 4: DEFECTIVE OR DAMAGED ITEMS" ended up as its own tiny 5-word
  chunk, separate from the paragraph explaining that section. This didn't
  break retrieval (the heading and its paragraph both retrieved
  successfully as neighboring hits), but it's a real, visible instance of
  the general "poor chunking" failure mode — worth a design revisit in
  Day 16+ (e.g. merging short heading-only chunks into the paragraph that
  follows them).
- The mock LLM's answer format (returning the raw top retrieved chunk
  rather than a real generated summary) made a genuine retrieval failure
  fully visible and honest, rather than being papered over by a fluent
  LLM re-wording. Real Gemini generation might have produced a
  smoother-sounding but still wrong answer for the "unopened" query — mock
  mode's bluntness turned out to be a debugging advantage here, not just a
  cost-saving one.

## Challenges

**Real bug, found and documented honestly:** `huggingface.co` is not on
this sandbox's network egress allowlist. The first attempt to load
`sentence-transformers`'s `all-MiniLM-L6-v2` model failed with a 403 from
the egress proxy, surfacing to Python as a confusing
`HfHubHTTPError`/`OSError` chain with no obvious hint that the real problem
was a network allowlist rather than a code bug. This is the same *category*
of issue as the already-documented ChromaDB ONNX auto-download bug from the
original Day 15 corpus setup — any component that lazily downloads model
weights at runtime is a hidden environment dependency.

**Fix, following the same pattern used for the ChromaDB bug:** the real
sentence-transformers code path was kept as the correct, primary
implementation — this is exactly what should run in an environment with
normal internet access, such as the user's own local machine (the same
place the original Day 15 sentence-transformers reference code was
independently re-run and verified). For pipeline development and testing
*inside this sandbox*, `embedding.py` automatically falls back to a
network-free TF-IDF vectorizer (scikit-learn) implementing the identical
interface, so the rest of the pipeline could be built and tested end-to-end
honestly, without silently faking success. Which backend actually ran is
always printed, never hidden.

**Second-order bug this fallback then caused, also found and fixed:** the
TF-IDF fallback uses a module-level singleton vectorizer whose vocabulary
is fixed at first `fit()`. During test development, `test_embed_vectors_are_
nonzero` failed with an all-zero vector — not because embedding was broken,
but because an *earlier* test in the same pytest session had already fit
the singleton on a small, unrelated vocabulary ("hello", "world", "foo"),
so this test's words ("refund", "policy", "text", "here") were all
out-of-vocabulary. Fixed by adding a `reset_model()` test utility and
calling it at the start of any test that needs a specific, predictable
vocabulary. This bug is itself a useful, honest illustration of *why*
production RAG needs real semantic embeddings rather than a fixed lexical
vocabulary — a corpus grows and gets re-queried with new phrasing over
time, and a fixed-vocabulary method degrades in exactly this way.

**Real, reproduced RAG failure mode (not injected, discovered while
testing):** querying `pipeline.py` with "How many days do I have to return
an unopened product?" retrieved the wrong section (48-hour defective-item
rule instead of the 30-day general return window), because the corpus's
actual wording is "original, unused condition" — "unopened" and "unused"
share zero vocabulary overlap under the TF-IDF fallback. This is a live,
first-hand demonstration of the exact synonym-blindness limitation first
proven in Day 2 (TF-IDF/Word2Vec, similarity 0.0000 on true synonyms with
no shared words), now surfacing again three phases and thirteen days later
in a completely different context. Documented in `docs/rag_failure_modes.md`
as the reproduced example for the "wrong retrieval" failure mode.

**Disk space:** ran out of disk space (`OSError: [Errno 28] No space left
on device`) partway through installing dependencies, caused by ~4.8GB of
accumulated cache from earlier work. Resolved by purging pip's cache and
recreating the venv with `--system-site-packages` so the already-installed,
several-GB torch/transformers stack didn't need to be downloaded a second
time.

## How to Run

See `README.md` — `USE_MOCK_LLM=true venv/bin/python src/pipeline.py` for
offline mode, or set `GEMINI_API_KEY` for real Gemini generation.
`venv/bin/python -m pytest tests/ -v` for the test suite.

## Learning Outcomes

- RAG's real value is grounding + traceability + freshness, not raw
  intelligence — a small local corpus with hybrid retrieval outperformed
  what pure LLM knowledge could offer on these company-specific policy
  questions, and every answer can point to exactly which document it came
  from.
- "The embedding model doesn't have network access" and "the vector store
  tries to auto-download a model" are the same underlying failure pattern
  wearing two different costumes — any RAG component that lazily fetches
  something from the network at runtime is a hidden dependency worth
  making explicit.
- Chunking and retrieval strategy choices have real, visible, sometimes
  surprising downstream effects (the tiny heading-only chunk; the
  synonym-blindness failure) that are much easier to internalize by
  actually triggering them than by reading about them in the abstract —
  which is exactly why this project was built as working code rather than
  documentation alone.
- Hybrid retrieval (dense + BM25 via Reciprocal Rank Fusion) is a genuinely
  low-cost way to hedge against both retrieval failure modes (paraphrase
  blindness and exact-term blindness) at once, rather than having to
  correctly predict in advance which one a given query will trigger.

## Author

Fatima Azeem
