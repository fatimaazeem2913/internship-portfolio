# RAG Evaluation Metrics (RAGAS)

RAGAS (Retrieval Augmented Generation Assessment) splits RAG evaluation
into two halves that matter separately: is the **retrieval** stage doing
its job (finding the right chunks), and is the **generation** stage doing
its job (answering faithfully and relevantly from what was found)? A
system can fail at either stage independently, so measuring only the final
answer's overall quality can hide *which* stage actually broke. Findings
below were verified via live web search rather than pulled from memory
alone — see Sources at the bottom.

| Metric | Stage measured | What it catches |
|---|---|---|
| Faithfulness | Generation | Hallucinated / unsupported claims |
| Answer Relevance | Generation | Answers that are technically true but off-topic |
| Context Precision | Retrieval | Irrelevant chunks diluting the retrieved set |
| Context Recall | Retrieval | Missing chunks — required info that never got retrieved |

## 1. Faithfulness

Faithfulness measures how much of the generated answer is actually
supported by the retrieved context, computed as the share of individual
claims in the answer that the context backs up:

```
Faithfulness = (number of claims in the answer supported by retrieved context) / (total number of claims in the answer)
```

This is the metric this project's `RAG_SYSTEM_PROMPT` (in `llm.py`) is
designed to protect: instructing the model to answer *only* from provided
context and to explicitly say when context is insufficient is a
faithfulness-preserving design choice, not just a nicety.

**Real-world failure this metric catches, but others don't:** one
documented case involved a legal research RAG system that shipped scoring
0.91 on faithfulness in its offline evaluation set. Three weeks into
production, users noticed roughly 1 in 6 responses was missing a key
statute — yet faithfulness stayed at 0.91 the entire time, because the
retriever was quietly missing a second required statute on multi-hop
questions, while the generator kept answering smoothly and faithfully from
the partial context it *did* have. The regression only became visible once
context recall was checked separately and turned out to have dropped to
0.62. This is exactly why faithfulness alone isn't a complete evaluation —
it can look perfect while the retrieval stage is quietly broken underneath.

## 2. Answer Relevance

Answer Relevance measures whether the generated answer actually addresses
the question that was asked, independent of whether it's factually
grounded — typically via semantic similarity between the question and the
answer. A high score means strong alignment with the question; a low score
means the answer drifted off-topic, stayed vague, or answered something
adjacent instead.

A faithful answer can still score low here: a response can be 100%
grounded in real retrieved text and still fail to address what was asked —
for example, faithfully summarizing the wrong retrieved section entirely.

## 3. Context Precision

Context Precision measures how much of what got retrieved was actually
useful. A high score means the retrieved set stayed tightly focused with
little irrelevant material mixed in:

```
Context Precision@k = (number of retrieved chunks in top-k that are actually relevant) / k
```

This is what this project's `top_k` tuning and hybrid RRF retrieval are
trying to protect: retrieving a small, well-focused set of chunks rather
than casting a wide net and hoping the LLM ignores the noise.

## 4. Context Recall

Context Recall measures the opposite failure direction: whether everything
that *should* have been retrieved actually was. A high score means the
necessary supporting evidence made it into the retrieved set; a low score
means something important got missed:

```
Context Recall = (number of ground-truth-relevant chunks that were retrieved) / (total number of ground-truth-relevant chunks that exist)
```

Context Recall requires labeled ground truth (knowing in advance which
chunks *should* have been retrieved for a given question) to compute
exactly — which is why, in practice, teams often only compute it on a
curated evaluation question set rather than for every live query.

## Why all four, not just one

Used together, these four metrics separate retrieval failures (irrelevant
or incomplete context) from generation failures (unsupported or
unresponsive answers) in a way no single overall score can. The legal-RAG
example above is the canonical illustration: a single high-level "is the
answer good" score would have missed the exact stage that broke.

## Applying this to `enterprise_rag_engine`

This project doesn't run automated RAGAS scoring (that requires either a
labeled ground-truth question set or an LLM-as-judge setup, both out of
scope for Day 15's theory-focused deliverable), but `tests/test_pipeline.py`
does the manual, hand-checked equivalent for a handful of known questions:
asserting the correct source file is present in `result["sources"]` is a
manual context-recall check, and asserting known-correct substrings appear
in mock answers is a manual faithfulness check. A natural Day 16+ extension
would be wiring up real RAGAS scoring against a small labeled eval set.

## Sources

- Ragas documentation, *List of available metrics* —
  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- ragR: *Retrieval-Augmented Generation and RAG Assessment in R* —
  https://arxiv.org/pdf/2604.23515
- FutureAGI, *RAG Evaluation Metrics in 2026: Faithfulness & More* —
  https://futureagi.com/blog/rag-evaluation-metrics-2025/
- *When Generic Prompt Improvements Hurt: Evaluation-Driven Iteration for
  LLM Applications* — https://arxiv.org/pdf/2601.22025
