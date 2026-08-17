"""
retrieval.py
------------
Vector Store -> Retrieval.

Implements three retrieval strategies, mirroring the three-way comparison
already done for the real Day 7 RAG mini-project:

  1. dense_retrieve()  -- pure semantic vector search via ChromaDB.
  2. bm25_retrieve()   -- pure lexical keyword search via rank-bm25.
  3. hybrid_retrieve()  -- reciprocal rank fusion (RRF) of both.

Why hybrid exists (ties directly to rag_failure_modes.md, "wrong retrieval"):
dense retrieval is strong at paraphrase/synonym matching but can miss exact
identifiers dense embeddings don't represent well -- section numbers, exact
dollar figures, SKUs, order-status keywords like "48 hours" as a literal
number. BM25 is the opposite: strong at exact term matches, blind to
paraphrase. A query like "How many hours do I have to report a broken
item?" needs dense retrieval's paraphrase understanding ("broken" ~
"defective"); a query like "SECTION 4" needs BM25's exact-match strength.
Hybrid retrieval covers both failure modes at once instead of picking one.

Reciprocal Rank Fusion (RRF) is used to combine the two ranked lists
because it doesn't require the two systems' raw scores (cosine distance
vs. BM25 score) to be on comparable scales -- it only uses each result's
*rank position* in each list, which sidesteps that scale mismatch entirely.
RRF formula: score(d) = sum over each ranker of  1 / (k + rank(d))
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from chunking import Chunk
from vector_store import VectorStore


class Retriever:
    def __init__(self, chunks: list[Chunk], vector_store: VectorStore):
        self.chunks = chunks
        self.chunk_by_id = {c.chunk_id: c for c in chunks}
        self.vector_store = vector_store

        # BM25 needs pre-tokenized text; simple whitespace/lowercase split is
        # sufficient for BM25's term-frequency statistics.
        self._tokenized_corpus = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def dense_retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        return self.vector_store.query(query, top_k=top_k)

    def bm25_retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]
        return [
            {"chunk_id": c.chunk_id, "text": c.text, "metadata": {"doc_id": c.doc_id, "source": c.source}, "score": float(s)}
            for c, s in ranked
        ]

    def hybrid_retrieve(self, query: str, top_k: int = 4, k_rrf: int = 60) -> list[dict]:
        dense_hits = self.dense_retrieve(query, top_k=max(top_k * 3, 10))
        bm25_hits = self.bm25_retrieve(query, top_k=max(top_k * 3, 10))

        rrf_scores: dict[str, float] = {}
        for rank, hit in enumerate(dense_hits):
            rrf_scores[hit["chunk_id"]] = rrf_scores.get(hit["chunk_id"], 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, hit in enumerate(bm25_hits):
            rrf_scores[hit["chunk_id"]] = rrf_scores.get(hit["chunk_id"], 0.0) + 1.0 / (k_rrf + rank + 1)

        ranked_ids = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        results = []
        for chunk_id, rrf_score in ranked_ids:
            chunk = self.chunk_by_id[chunk_id]
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk.text,
                    "metadata": {"doc_id": chunk.doc_id, "source": chunk.source},
                    "rrf_score": rrf_score,
                }
            )
        return results


if __name__ == "__main__":
    from ingestion import load_corpus
    from chunking import chunk_documents

    docs = load_corpus("data/corpus")
    chunks = chunk_documents(docs)
    store = VectorStore(persist_dir="data/vector_store_test")
    store.reset()
    store.add_chunks(chunks)

    retriever = Retriever(chunks, store)

    # A query designed to need exact-term matching (BM25's strength) --
    # "SECTION 5" is a literal heading, not a paraphrasable concept.
    query = "SECTION 5 late refunds"
    print(f"Query: {query!r}\n")

    print("-- Dense only --")
    for r in retriever.dense_retrieve(query, top_k=2):
        print(f"  [{r['chunk_id']}] {r['text'][:80].strip()}...")

    print("\n-- BM25 only --")
    for r in retriever.bm25_retrieve(query, top_k=2):
        print(f"  [{r['chunk_id']}] score={r['score']:.3f} {r['text'][:80].strip()}...")

    print("\n-- Hybrid (RRF) --")
    for r in retriever.hybrid_retrieve(query, top_k=2):
        print(f"  [{r['chunk_id']}] rrf={r['rrf_score']:.4f} {r['text'][:80].strip()}...")
