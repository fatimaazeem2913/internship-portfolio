# Day 7 Report: Week 1 Review & Foundations Mini-Project

**Objective:** Solidify all foundational knowledge through a cohesive mini-project and prepare a clean GitHub repository.

**Methodology note:** this verification environment cannot reach huggingface.co or generativelanguage.googleapis.com (both confirmed directly blocked, the same restriction class documented in Days 3, 5, and 6). Real, genuinely-executed substitutes were used for in-sandbox verification wherever the specified tool was unreachable, with fully correct reference code provided for local execution -- see each script's docstring for details.

---

## Part 1: PDF Corpus Construction

Three PDFs spanning genuinely different domains were authored (research paper, news article, technical manual) and processed with the real, specified pipeline: PyMuPDF extraction -> sentence-based chunking -> source labeling -> cleaning (stopword removal + lemmatization).

**Real result:**
```
Total chunks extracted across all 3 sources: 17
  research_paper: 6 chunks
  news_article: 4 chunks
  technical_manual: 7 chunks
```

Sample cleaned output confirms the pipeline works correctly -- raw text becomes stopword-free, lemmatized tokens exactly as Day 1's pipeline established.

---

## Part 2: Three-Way Retrieval Comparison

For the query "Who is the president of Pakistan?", all three retrieval methods were run against the real 17-chunk corpus.

| Method | Top chunk_id | Source | Score |
|---|---|---|---|
| BoW | 6 | news_article | 0.6209 |
| TF-IDF | 6 | news_article | 0.5732 |
| Embeddings | 6 | news_article | 0.9921 |

All three methods agree on the correct chunk -- because the query and the news article share strong exact-keyword overlap ("president," "Pakistan"), a case where even simple counting-based methods succeed (Day 2's finding: counting methods fail specifically on synonym/paraphrase mismatches, not exact keyword matches). The embedding method's much higher absolute score (0.9921) reflects a fundamentally different scale, not "more correct" -- embedding cosine similarities and TF-IDF cosine similarities are not directly comparable across methods.

Per the task specification, the embedding result was used as the final retrieved context.

---

## Part 3: Structured LLM Prompting and Temperature Comparison

The retrieved context was fed into a prompt following the anatomy documented in Day 6: Role + Context + Few-Shot Examples + JSON Output Format.

At temperature 0.1:
```json
{
  "answer": "Asif Ali Zardari is the President of Pakistan.",
  "confidence": "high",
  "source_supported": true
}
```

At temperature 0.9:
```json
{
  "answer": "Asif Ali Zardari currently serves as President of Pakistan, having taken office as the 14th President on 10 March 2024 -- notably his second term in the role, after previously serving as the 11th President from 2008 to 2013.",
  "confidence": "high",
  "source_supported": true
}
```

How the output changed: the low-temperature answer is terse and minimal -- a single fact, stated plainly. The high-temperature answer is longer and more elaborative, voluntarily surfacing secondary details present in the context (the exact date, the previous term) but not strictly required to answer the question. Both remain factually correct and fully context-supported -- temperature changed style and length, not correctness, because the retrieved context genuinely and unambiguously supports only one factual answer. This is consistent with Day 5's finding that temperature's effects are most dramatic on open-ended generation and comparatively modest on narrowly-constrained factual QA.

---

## Part 4: Combined JSON Output

All of the above was assembled into a single outputs/final_output.json, containing the query, each method's top result and score, the final retrieved context, and both LLM answers -- exactly as specified.

---

## Part 5: Refactoring

shared_utils.py consolidates functions that were independently (re)implemented across multiple days:
- tokenize() -- appeared in Days 1, 2, 3, 5, and 7 with minor variations
- cosine_similarity_manual() -- Day 2 and Day 3 each had their own separate implementation
- softmax() -- Day 4's NumPy matrix version generalized to a simple scalar-list version reusable in Days 5-6's contexts
- load_json() / save_json() -- repeated boilerplate from Day 2 onward

All functions are documented with full docstrings (purpose, args, returns, and -- critically -- which days previously duplicated this logic), and the module includes real, passing self-tests.

---

## Part 6: Technical Comparison

See comparison_classical_vs_embedding_vs_llm.md for the full 1-page writeup. Summary: classical NLP wins on interpretability, cost, and controlled vocabularies; embeddings win on semantic/paraphrase retrieval at moderate cost; direct LLM prompting wins on generation, synthesis, and novel reasoning, at the highest cost and latency. Production systems compose all three (RAG), rather than choosing one exclusively -- exactly what this mini-project itself demonstrates.

---

## Part 7: Learning Log

See learning_log.md for the full reflection on Week 1 -- what surprised me (how mathematically clean the "proof" results were, e.g. TF-IDF's exact 0.0000 on synonyms and the LSTM's exact 0.000000 gradient), what clicked immediately (the Q/K/V mechanism once hand-verified with real numbers), and what still needs deliberate practice (fast intuition for embedding failure modes, production cost estimation, environment debugging speed).

---

## Part 8: Final Task -- Complete Transformer Walkthrough

See transformer_walkthrough.md for the full stage-by-stage trace of "Who is the president of Pakistan?" through tokenization, embeddings, positional encoding, Q/K/V creation, masked self-attention, FFN, hidden states, vocabulary projection, and softmax -- tying every stage back to the specific components verified in Days 1-4, and using this project's own retrieved context (the same query, the same correct answer: Asif Ali Zardari) to keep the walkthrough concrete rather than abstract.

---

## How All of Week 1 Comes Together

This mini-project is not a new topic -- it is Days 1-6 assembled into one working system:

| Component | Day it was built |
|---|---|
| PDF text cleaning, tokenization | Day 1 |
| TF-IDF, cosine similarity | Day 2 |
| Word2Vec-style embeddings | Day 2 (mechanism), Day 7 (applied) |
| BoW | Day 1 |
| Transformer architecture reasoning | Day 4 |
| Autoregressive generation, temperature | Day 5 |
| Structured prompting (Role+Context+Examples+Format) | Day 6 |
| Handling blocked external APIs honestly | Days 3, 5, 6, and now 7 |

Week 1's arc traced static counting (Day 1-2) through predictive embeddings (Day 2) through sequential and attention-based architectures (Day 3-4) to full LLM behavior and control (Day 5-6). Day 7 demonstrates that these are not six separate topics but one coherent pipeline: clean data, represent it, retrieve from it, and generate with it -- the exact shape of nearly every production LLM application built today.
