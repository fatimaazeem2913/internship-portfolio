# RAG Failure Modes & Fixes

Four failure modes, each reproduced (not just described) in this project's
`enterprise_rag_engine/`.

## 1. Poor Chunking

**What goes wrong:** splitting a document at a fixed character count, with
no regard for sentence or paragraph boundaries, can cut a rule apart from
the sentence that qualifies it. For example, splitting mid-sentence between
"Refunds are processed within 5-7 business days" and the very next sentence
about what happens if the original payment method is no longer valid would
retrieve an *incomplete* rule — technically true, but missing a caveat that
changes the real answer.

**Fix applied in this project (`src/chunking.py`):** paragraph-aware
recursive splitting. Chunks are split on paragraph boundaries first; only a
paragraph that's still too long falls back to a word-count sliding window,
and even then with configurable overlap (default 20 words) so a rule and
its immediately following qualifier are unlikely to be fully separated.

**Real, reproduced example from this project:** the pipeline's own mock-mode
test run for the query *"How many days do I have to return an unopened
product?"* retrieved the wrong section (the 48-hour defective-item rule
instead of the 30-day general return window) — not because of bad
chunking, but because of failure mode #2 below. This is exactly the kind
of cross-cutting failure real RAG systems hit: multiple failure modes can
produce the same symptom (wrong answer), so fixing one doesn't guarantee
the others aren't also present.

## 2. Wrong Retrieval

**What goes wrong:** the retrieval method doesn't match the query's needs.
Pure semantic (dense) retrieval can miss queries that hinge on exact terms
or identifiers; pure lexical (BM25) retrieval misses queries phrased as a
paraphrase with no shared vocabulary with the source text.

**Real, reproduced example from this project:** because this sandbox has
no network access to huggingface.co (see `embedding.py` docstring), the
embedding fallback used here is TF-IDF — a purely lexical method. Querying
*"How many days do I have to return an unopened product?"* against a
corpus whose actual wording is *"original, unused condition"* returned the
wrong section, because "unopened" and "unused" share zero vocabulary
overlap. A real semantic embedding model (the intended production path)
would correctly recognize these as near-synonyms — this is the exact
synonym-blindness limitation of lexical methods first proven in Day 2
(TF-IDF/Word2Vec similarity of 0.0000 on true synonyms with no shared
words).

**Fix applied in this project (`src/retrieval.py`):** hybrid retrieval —
dense + BM25 combined via Reciprocal Rank Fusion — so a query only needs to
succeed under *either* method, not both, to surface the right chunk. In
production with real sentence-transformers embeddings (rather than this
sandbox's TF-IDF fallback), the dense half of the hybrid search would
correctly catch the "unopened"/"unused" paraphrase that TF-IDF alone missed.

## 3. Context Overflow

**What goes wrong:** retrieving too many chunks, or chunks that are too
large, exceeds the LLM's context window or crowds out the actual question
among a wall of retrieved text — degrading answer quality even when the
right information technically was retrieved (the "needle in a haystack"
problem: LLMs answer less reliably when the relevant fact is buried in a
lot of irrelevant surrounding context).

**Fix applied in this project:** `top_k` is a configurable, deliberately
small default (4) in `pipeline.py`/`vector_store.py`, rather than
retrieving "as many as might be relevant." Combined with paragraph-aware
chunking (chunks average well under 120 words), a full retrieval result set
stays small enough to leave the model's context window mostly free for
reasoning, not just reading.

## 4. Hallucination Despite Retrieval

**What goes wrong:** even with the right chunks retrieved, an LLM can still
ignore them and answer from its own pretrained knowledge, or blend
retrieved facts with invented ones — especially on questions the retrieved
context only partially answers.

**Fix applied in this project (`src/llm.py`, `RAG_SYSTEM_PROMPT`):** the
system prompt explicitly instructs the model to answer *only* from
provided context, to say so explicitly when the context is insufficient
rather than guess, and to cite which source it used. `generate_answer()`
with an empty retrieval result set is a hard-coded honest refusal ("I don't
have enough information..."), verified by
`tests/test_pipeline.py::test_generate_answer_with_no_chunks_says_so`,
rather than letting the LLM call happen anyway and risk answering from
outside knowledge with nothing to ground it. RAGAS's **Faithfulness**
metric (see `rag_evaluation.md`) is the standard way to measure this failure
mode quantitatively rather than spot-checking by hand.
