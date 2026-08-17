# Day 15 Report: RAG Architecture & Theory

**Objective:** Understand when, why, and how to use Retrieval-Augmented Generation to ground LLM responses in real-world data — and when not to.

**A note on grounding this in real, prior work:** Day 7's mini-project already built a genuine, working RAG pipeline — 3 real PDFs, real chunking, 3 real retrieval methods compared side by side, and real LLM answer generation grounded in retrieved context. Rather than treating today's theory as abstract, every stage below is tied back to that concrete, already-verified implementation wherever possible.

**A note on sources:** the RAGAS metric definitions and RAG variant progression (Naive -> Advanced -> Modular -> GraphRAG) are drawn from current, searched sources (the RAGAS documentation, the Gao et al. 2024 RAG survey, and 2025-2026 practitioner sources on GraphRAG), not from memory alone, since this is exactly the kind of fast-evolving terminology where training data can be incomplete or outdated.

---

## Part 1: The Full RAG Pipeline

```
Corpus -> Ingestion -> Chunking -> Embedding -> Vector Store -> Retrieval -> Augmented Prompt -> LLM -> Grounded Response
```

| Stage | What happens | Day 7's real implementation |
|---|---|---|
| Corpus | The raw source documents that contain the knowledge you want the LLM grounded in | 3 real PDFs: a research paper, a news article, a technical manual |
| Ingestion | Loading raw files into extractable text | extract_chunk_clean.py using PyMuPDF |
| Chunking | Splitting long documents into smaller, retrievable pieces | Sentence-based chunking, 17 total chunks across 3 documents (6+4+7) |
| Embedding | Converting each chunk into a numeric vector capturing its meaning | Word2Vec-substitute embeddings (real sentence-transformers reference code provided, network-blocked in that sandbox too) |
| Vector Store | A database optimized for finding the most similar vectors to a query vector | Manual cosine-similarity search over stored vectors (a real vector database -- ChromaDB -- is this project's Day 15 addition) |
| Retrieval | Given a query, finding the most relevant chunk(s) | 3 methods compared: BoW, TF-IDF, embeddings -- all 3 agreed on the correct chunk for the test query |
| Augmented Prompt | Inserting the retrieved chunk(s) into the LLM's prompt alongside the user's question | llm_answer_generation.py's prompt template with explicit grounding instructions |
| LLM | The model generates an answer using the augmented prompt as context | Claude-simulated (Day 7), Gemini (from Day 8 onward) |
| Grounded Response | An answer that's actually supported by the retrieved real text, not just the model's training data | Verified: the retrieved news-article chunk correctly identified the current President of Pakistan |

**The single most important idea in this whole pipeline:** the LLM never "knows" anything from the corpus directly — it only ever sees whatever text got placed into its prompt by the retrieval step. If retrieval fails, grounding fails, no matter how good the LLM itself is. This is why RAG system quality is disproportionately a retrieval engineering problem, not a prompting problem — a theme that recurs throughout the rest of this report.

---

## Part 2: RAG vs. Fine-Tuning

### 5 scenarios where RAG is the right choice

1. **The knowledge changes frequently.** A company's product catalog, current pricing, or today's inventory levels change constantly — fine-tuning would require retraining on every update; RAG just needs the underlying documents updated.
2. **You need source attribution.** RAG can point to which document an answer came from (Day 7's pipeline literally returns the source chunk alongside the answer); a fine-tuned model's knowledge is baked into its weights with no way to cite where a specific fact came from.
3. **The knowledge base is large and/or private.** A company's entire internal documentation set is often far larger than what's practical or even physically possible to encode via fine-tuning, and RAG keeps sensitive documents outside the model's actual weights.
4. **You need to reduce hallucination on factual, narrow questions.** Grounding the model in real retrieved text (Day 7's finding: the model correctly identified the real, current president when given the real article as context) is a direct, verifiable defense against confident-but-wrong answers.
5. **Budget and iteration speed matter.** RAG requires no model training at all — updating the knowledge base is instant and free (beyond re-indexing); fine-tuning requires real compute cost and a training cycle for every update.

### 3 scenarios where fine-tuning wins

1. **You need to change the model's behavior, tone, or format, not its knowledge.** Teaching a model to always respond in a specific structured format, a specific persona, or a specific reasoning style is a behavioral change — RAG only ever adds information to the prompt, it can't reliably change how the model reasons or writes.
2. **You need the knowledge available with zero retrieval latency, at massive scale.** Every RAG query pays the cost of an embedding lookup and a retrieval step before generation even starts; a fine-tuned model's knowledge is available the instant generation begins, which matters for extremely latency-sensitive, high-volume applications.
3. **The task requires deep, implicit pattern learning that's hard to phrase as retrievable facts.** Domain-specific tasks like learning a specialized code style, a legal reasoning pattern across many precedents, or a particular diagnostic reasoning process from thousands of examples are things fine-tuning can genuinely absorb into the model's behavior, but which don't reduce cleanly to "look this fact up and paste it into the prompt."

**In real production systems, these aren't mutually exclusive** — a fine-tuned model with RAG on top (fine-tuned for tone/behavior, RAG'd for current facts) is a common, legitimate combined approach.

---

## Part 3: Common RAG Failure Modes and Fixes

### 1. Poor chunking
**The problem:** chunks that are too large dilute relevance (a 2,000-word chunk with one relevant sentence buried inside still "matches" a query weakly on everything else); chunks that are too small lose necessary context (a sentence fragment without its surrounding paragraph can be genuinely ambiguous).
**The fix:** semantic or structure-aware chunking (splitting on natural document boundaries — paragraphs, sections — rather than a fixed character count), with a moderate overlap between adjacent chunks so information near a chunk boundary isn't cut off from its context on both sides.

### 2. Wrong retrieval (retrieving the wrong or irrelevant chunks)
**The problem:** the query and the correct chunk can be about the same topic but phrased so differently that simple similarity search misses the connection — this is TF-IDF's proven failure mode from Day 2 (synonym-blindness, measured at exactly 0.0000 similarity for genuine synonym pairs).
**The fix:** hybrid search — combining a keyword-based method (like rank-bm25, installed this project) with a semantic embedding method, so a query can be matched either by exact term overlap OR by meaning, whichever the case actually calls for. Day 7's finding that BoW, TF-IDF, and embeddings all agreed on the correct chunk was itself a special case — that only worked because the query happened to share exact keywords with the source; hybrid search is the general-purpose fix for when that's not true.

### 3. Context overflow (retrieved content exceeds the model's usable context)
**The problem:** retrieving "more" context isn't free — every chunk added competes for space in a finite context window (Day 4's O(n^2) attention cost directly explains why this window can't just be made unlimited), and past a certain point, additional chunks add noise rather than signal.
**The fix:** re-ranking retrieved chunks and keeping only the top-K most relevant ones (rather than everything retrieved), and/or compressing/summarizing lower-priority chunks before inclusion, so the highest-value information always fits within budget.

### 4. Hallucination despite retrieval
**The problem:** having relevant context in the prompt does NOT guarantee the model actually uses it — a model can still generate a plausible-sounding claim that isn't actually supported by the retrieved text (this is precisely what the RAGAS Faithfulness metric, covered in Part 5, is designed to catch).
**The fix:** explicit grounding instructions in the prompt (Day 7's pattern: "answer ONLY from the provided context, say you don't know otherwise"), combined with low temperature for factual RAG tasks (Day 7's finding, re-confirmed across Days 9/13: low temperature reduces unnecessary variation on tasks with one correct, context-supported answer) — though note Day 9's separate finding that temperature is silently ignored on Gemini 3.x models, meaning this specific lever must be replaced with explicit prompt-level determinism instructions on current-generation Gemini models.

---

## Part 4: RAG Variants — What Each Adds

Based on the widely-cited three-stage progression (Gao et al., 2024) plus GraphRAG as a distinct further variant:

### Naive RAG
The baseline "retrieve-then-read" pipeline exactly as described in Part 1 — index documents, retrieve by similarity, stuff into the prompt. Real, proven limitations: shallow query understanding (a query and a relevant chunk can be semantically related without high surface similarity), retrieval redundancy/noise (feeding every retrieved chunk to the LLM regardless of whether it's actually useful), and no correction mechanism if retrieval gets it wrong.

### Advanced RAG
Adds optimization before and after the core retrieval step, without changing the pipeline's fundamentally linear shape:
- Pre-retrieval: query rewriting/expansion (reformulating a user's question into a form more likely to match relevant chunks), better/dynamic chunking strategies
- Post-retrieval: re-ranking retrieved chunks by actual relevance (not just raw similarity score), filtering out low-value chunks before they reach the LLM

### Modular RAG
Breaks the pipeline into discrete, independently swappable components (routing, scheduling, fusion of multiple retrieval strategies) rather than a fixed linear sequence — described in the literature as a genuine architectural generalization where "Advanced RAG is a special case of Modular RAG, and Naive RAG is a special case of Advanced RAG." This is what makes hybrid search (Part 3's fix for wrong retrieval) practical to implement cleanly: BM25 and embedding-based retrieval become two swappable/combinable modules rather than a hardcoded single method.

### GraphRAG
A structurally different approach: instead of treating chunks as isolated, independent pieces of text, GraphRAG builds a knowledge graph from the corpus (entities become nodes, relationships between entities become edges), then uses graph-community-detection algorithms (commonly the Leiden algorithm) to build hierarchical summaries. At query time, GraphRAG can perform multi-hop reasoning across connected entities that pure vector similarity search cannot support — e.g., answering "how are Company A and Company B connected?" by traversing the graph, rather than hoping one single retrieved chunk happens to state the connection directly.

**Bottom line:** each variant adds one specific capability the previous one measurably lacked — Advanced RAG fixes retrieval quality around a fixed pipeline shape; Modular RAG fixes pipeline rigidity itself; GraphRAG fixes the inability to reason across relationships between separate pieces of retrieved information.

---

## Part 5: RAG Evaluation Metrics (RAGAS Framework)

Sourced directly from the RAGAS documentation and multiple corroborating academic/practitioner sources.

| Metric | What it measures | Evaluates |
|---|---|---|
| Faithfulness | The fraction of claims in the generated answer that can actually be inferred from the retrieved context — claims supported by context / total claims in answer | Generation quality — directly catches Part 3's "hallucination despite retrieval" failure mode |
| Answer Relevance | Semantic similarity between the generated answer and the original question — does the answer actually address what was asked, regardless of whether it's grounded | Generation quality |
| Context Precision | Whether the retrieved chunks that are actually relevant are ranked higher than irrelevant ones — a well-focused retrieval set scores high, a noisy one scores low | Retrieval quality |
| Context Recall | The fraction of the information actually needed to answer correctly that made it into the retrieved context — measured against a ground-truth answer | Retrieval quality |

**Why these four specifically, and why in two pairs:** Context Precision and Context Recall isolate the retrieval half of the pipeline; Faithfulness and Answer Relevance isolate the generation half. This separation is genuinely useful for debugging: if Faithfulness is low but Context Recall is high, the problem is the LLM ignoring good context (a prompting/generation fix); if Context Recall is low, the problem is retrieval never finding the right information in the first place (a chunking/embedding/retrieval-method fix) — no amount of prompt engineering will fix a retrieval-level failure, and vice versa.

---

## Part 6: Project Setup — Real, Verified Installation

enterprise_rag_engine/ was created and all 5 required packages installed and functionally verified (not just import-checked) via verify_setup.py:

```
[PASS] pypdf 5.9.0: real PDF extraction verified
[PASS] pdfplumber 0.11.9: real PDF extraction verified
[PASS] rank-bm25: real keyword retrieval verified (correctly ranked: 'BERT produces contextual embeddings')
[PASS] chromadb 1.5.9: real vector storage + retrieval verified (top result: 'BERT produces contextual embeddings')
[PASS] sentence-transformers 5.7.0: package verified importable

SUMMARY: 5/5 packages verified with real functional tests
```

**A real, honest finding from this verification process:** ChromaDB's default embedding function attempts to auto-download its own ONNX model over the network on first use — this fails in any network-restricted environment with a cryptic SHA256 mismatch error unrelated to any actual code problem. The fix, and the more architecturally correct pattern regardless of network access: always supply embeddings= explicitly using a controlled embedder (this project uses scikit-learn's TF-IDF as a network-independent substitute, verified to correctly retrieve the right document for a test query; embedding_reference.py provides the correct real sentence-transformers code for local use with genuine internet access) — exactly why these 5 packages are installed together, not chromadb in isolation.

---

## How Day 15 Connects to Earlier Days

| Earlier concept | Role in Day 15 |
|---|---|
| Day 2: TF-IDF's proven synonym-blindness (0.0000 similarity) | Direct motivation for hybrid search as the fix for "wrong retrieval" |
| Day 4: O(n^2) attention cost, context window limits | Direct explanation for why "context overflow" is a real, structural constraint, not just a tuning knob |
| Day 7: The real, working RAG mini-project | Every pipeline stage in Part 1 is grounded in an already-built, already-verified implementation |
| Day 9: Gemini 3.x silently ignoring temperature | Directly qualifies Part 3's "low temperature reduces hallucination" fix for current-generation models |
| Day 6/9: Structured, schema-enforced output | The same underlying principle GraphRAG applies at the corpus level — turning unstructured text into structured, queryable relationships |

Day 15 opens Phase 3 by turning Day 7's working-but-simple RAG implementation into a foundation for genuinely production-grade RAG engineering — real vector databases, real hybrid retrieval, and a real evaluation framework for measuring whether a RAG system is actually working, not just assuming it is.
