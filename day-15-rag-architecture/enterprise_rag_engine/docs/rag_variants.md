# RAG Variants: Naive → Advanced → Modular → GraphRAG

Based on the widely-cited RAG survey framework (Gao et al., 2024,
*"Retrieval-Augmented Generation for Large Language Models: A Survey"*,
arXiv:2312.10997), which categorizes RAG's evolution into three paradigms,
plus GraphRAG as a further structural departure. Findings below were
verified via live web search rather than pulled from memory alone, since
this is a fast-moving area — see Sources at the bottom.

## Naive RAG

The original, simplest pattern, and the exact 3-step shape this project's
`pipeline.py` implements at its core: documents are split into chunks and
encoded into vectors during indexing; at query time, the top-k most similar
chunks are retrieved; the question plus those chunks are handed to the LLM
to produce the final answer.

**Limitation:** early Naive RAG implementations leaned on sparse, keyword-
overlap retrieval (like plain BM25), which gives limited semantic
understanding — often surfacing noisy or redundant chunks and struggling to
connect information that's spread across multiple sources.

## Advanced RAG

Advanced RAG layers targeted improvements onto Naive RAG's weak points,
mainly around retrieval quality: pre-retrieval steps like query rewriting
or expansion, and post-retrieval steps like re-ranking results or filtering
by metadata, plus more careful chunking strategies upstream.

**What this project does that maps to Advanced RAG:** the hybrid dense+BM25
retrieval with Reciprocal Rank Fusion in `retrieval.py` is a post-retrieval
enhancement in this spirit — combining two rankers rather than trusting a
single retriever's raw output.

## Modular RAG

Modular RAG goes a step further and breaks the whole indexing → retrieval →
generation pipeline into separate, swappable components, each of which can
be improved, replaced, or reconfigured independently. This flexibility is
part of why it's become the dominant real-world pattern — it supports both
straightforward linear pipelines and more complex, iterative setups.

**What this project does that maps to Modular RAG:** the entire
`enterprise_rag_engine/src/` layout — `ingestion.py`, `chunking.py`,
`embedding.py`, `vector_store.py`, `retrieval.py`, `llm.py` as separate,
independently-testable modules wired together by `pipeline.py` — is a small
real instance of this pattern. `pipeline.py`'s `retrieval_mode` parameter
(`"dense"` / `"bm25"` / `"hybrid"`) is a direct, swappable-module example.

## GraphRAG

A structural departure rather than just a refinement. Instead of embedding
flat document chunks, GraphRAG first extracts entities and the relationships
between them into a knowledge graph. Retrieval then becomes graph traversal
(often over graph "communities" found via clustering algorithms like
Leiden) rather than nearest-neighbor vector search, and the LLM synthesizes
an answer from the relevant graph structure it finds.

**When it's worth it:** GraphRAG earns its extra complexity on queries that
hinge on relationships spanning a large document collection — compliance
analysis, research synthesis, competitive intelligence — where multi-hop
connections matter. For plain factual lookup, ordinary vector-based RAG
stays faster, cheaper, and just as accurate.

This project's corpus (two short policy documents) has no meaningful
entity/relationship graph to build — GraphRAG would be pure overhead here,
which is itself an honest illustration of the "when NOT to use it" half of
this project's objective.

## Beyond the original three: what's emerged since (2026)

Two further patterns have become common enough to be worth knowing, even
though they sit outside the original Naive/Advanced/Modular/GraphRAG
framework:

- **Agentic RAG** — the retrieval step itself becomes multi-step and
  tool-using: the model can decide to search again, refine its query, or
  call other tools, rather than a single fixed retrieve-then-generate pass.
- **Adaptive RAG** — a query-complexity classifier routes each query to the
  right pipeline automatically: simple factual questions go to fast, cheap
  Naive/Advanced RAG; complex multi-hop questions get routed to the slower,
  more thorough Agentic RAG path; relationship-heavy questions route to
  GraphRAG.

This project's `retrieval_mode` parameter is a manual, developer-set version
of that same routing idea, just without an automatic classifier choosing it.

## Sources

- Gao et al., *Retrieval-Augmented Generation for Large Language Models: A
  Survey* — https://arxiv.org/pdf/2312.10997
- *Graph Retrieval-Augmented Generation: A Survey* —
  https://arxiv.org/pdf/2408.08921
- *RAG Techniques Compared: A Practical Guide to Retrieval Augmented
  Generation in 2026* — https://blog.starmorph.com/blog/rag-techniques-compared-best-practices-guide
