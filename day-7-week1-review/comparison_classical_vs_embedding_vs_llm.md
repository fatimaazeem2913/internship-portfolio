# Technical Comparison: Classical NLP vs. Embedding Approach vs. Direct LLM Prompting

## The Three Approaches, Side by Side

| Dimension | Classical NLP (BoW/TF-IDF) | Embedding Approach (Word2Vec/Sentence-Transformers) | Direct LLM Prompting |
|---|---|---|---|
| **How it works** | Counts word occurrences, weights by rarity | Learns/uses dense vectors capturing meaning; ranks by cosine similarity | Sends the full query (and optionally context) directly to a generative model |
| **Understands synonyms?** | No — proven in Day 2 (0.0000 similarity on synonym pairs) | Yes, partially (Day 2: 0.34-0.50) to well (sentence-transformers, trained on billions of sentence pairs) | Yes — full semantic and contextual understanding |
| **Understands context/polysemy?** | No | No (static, Day 2) / Yes (contextual, e.g. BERT-derived) | Yes — fully contextual, computed per-query |
| **Setup cost** | Minimal — a few lines of sklearn | Moderate — needs a trained/pretrained model | Minimal for the call itself, but requires prompt engineering (Day 6) for reliability |
| **Latency** | Very fast (millisecond-scale, no GPU) | Fast (embedding lookup + cosine similarity) | Slower — full generative forward pass, sequential token-by-token (Day 5) |
| **Cost per query** | Effectively free | Cheap (embedding APIs are typically far cheaper than generation APIs) | More expensive — priced per token, both input and output |
| **Interpretability** | Fully transparent — every score traces to exact word overlap | Less transparent — similarity is geometric, not directly inspectable | Least transparent — output is a generated sequence, not a traceable score |
| **Deterministic?** | Yes, always | Yes (embeddings are fixed given fixed input) | Only at temperature 0; otherwise stochastic (Day 5/6) |

## When to Use Each

**Classical NLP (BoW/TF-IDF)** is the right choice when: the vocabulary is controlled and consistent (e.g., searching a codebase, matching exact product SKUs), you need full interpretability (a compliance system that must explain *why* a document matched), or you're operating at a scale/cost point where even embedding inference is too expensive. Day 7's own retrieval comparison showed BoW and TF-IDF correctly identified the right chunk *when the query shared exact keywords with the source* — this is the case classical NLP handles well. It fails, provably, the moment a query and its relevant document use different words for the same concept (Day 2's synonym-blindness proof).

**The Embedding Approach** is the right choice for semantic search and retrieval at scale: FAQ matching, document retrieval for RAG pipelines (exactly Day 7's Stage 2), deduplication, and clustering. It's the correct middle ground when you need to understand *meaning* but don't need to *generate* new text — retrieval, not composition, is the task. It is measurably better than classical NLP at handling paraphrase and synonymy (Day 2's Word2Vec results), while remaining far cheaper and faster than a full LLM call.

**Direct LLM Prompting** is the right choice when the task requires *generation*, multi-step reasoning, synthesis across multiple facts, or handling genuinely novel phrasing/instructions the other two approaches have no mechanism for. It is the only one of the three that can produce a *new*, context-aware, natural-language answer rather than merely *locating* existing text. It is also the most expensive and slowest per-query, and — without careful prompt engineering (Day 6) — the least reliable in terms of consistent output format.

## The Real-World Pattern: They Compose, They Don't Compete

Day 7's own mini-project demonstrates the actual production pattern used across the industry: **retrieval-augmented generation (RAG)**. Classical NLP or embeddings are not typically used as the *final* answer — they are used to *narrow down* a large corpus to the single most relevant piece of context (Stage 2 of this project), which is then handed to an LLM (Stage 3) to synthesize into a final, natural-language, directly-responsive answer. This is a deliberate, cost-aware pipeline: the cheap, fast methods do the expensive part (searching potentially millions of documents), and the expensive, slow method (the LLM) only ever has to process one short, already-relevant piece of text — not the entire corpus.

**The escalation principle**, consistent with Day 6's prompting-vs-RAG-vs-fine-tuning framework: try classical NLP first if the task is simple keyword matching; escalate to embeddings when synonymy/paraphrase matters; reserve full LLM generation for the step that genuinely requires producing new text, not merely finding existing text — since that step is the most expensive and slowest of the three at every stage of this pipeline.
