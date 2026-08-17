"""
pipeline.py
-----------
Wires the full RAG pipeline end to end:

  Corpus -> Ingestion -> Chunking -> Embedding -> Vector Store
         -> Retrieval -> Augmented Prompt -> LLM -> Grounded Response

This is the runnable implementation of the pipeline diagram documented in
docs/RAG_STUDY.md. Every stage in this file corresponds 1:1 to a stage in
that diagram and to a real module: ingestion.py, chunking.py, embedding.py,
vector_store.py, retrieval.py, llm.py.
"""

from __future__ import annotations

from chunking import chunk_documents
from ingestion import load_corpus
from llm import generate_answer
from retrieval import Retriever
from vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/vector_store",
        chunk_size: int = 120,
        chunk_overlap: int = 20,
        retrieval_mode: str = "hybrid",  # "dense" | "bm25" | "hybrid"
        top_k: int = 4,
    ):
        self.corpus_dir = corpus_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k

        self.store = VectorStore(persist_dir=persist_dir)
        self.retriever: Retriever | None = None
        self.chunks = []

    def build_index(self, reset: bool = True) -> int:
        """Corpus -> Ingestion -> Chunking -> Embedding -> Vector Store."""
        docs = load_corpus(self.corpus_dir)
        self.chunks = chunk_documents(docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap)

        if reset:
            self.store.reset()
        self.store.add_chunks(self.chunks)
        self.retriever = Retriever(self.chunks, self.store)
        return len(self.chunks)

    def query(self, question: str, top_k: int | None = None) -> dict:
        """Retrieval -> Augmented Prompt -> LLM -> Grounded Response."""
        if self.retriever is None:
            raise RuntimeError("Index not built yet -- call build_index() first.")

        k = top_k or self.top_k
        if self.retrieval_mode == "dense":
            hits = self.retriever.dense_retrieve(question, top_k=k)
        elif self.retrieval_mode == "bm25":
            hits = self.retriever.bm25_retrieve(question, top_k=k)
        else:
            hits = self.retriever.hybrid_retrieve(question, top_k=k)

        result = generate_answer(question, hits)
        result["retrieved_chunks"] = hits
        result["retrieval_mode"] = self.retrieval_mode
        return result


if __name__ == "__main__":
    import os

    os.environ.setdefault("USE_MOCK_LLM", "true")

    pipeline = RAGPipeline(retrieval_mode="hybrid", top_k=3)
    n_chunks = pipeline.build_index()
    print(f"Indexed {n_chunks} chunks from {pipeline.corpus_dir}\n")

    test_questions = [
        "How many days do I have to return an unopened product?",
        "What happens if my refund doesn't show up after two weeks?",
        "Can I get free overnight shipping?",
    ]

    for q in test_questions:
        result = pipeline.query(q)
        print(f"Q: {q}")
        print(f"   Sources: {result['sources']}")
        print(f"   A: {result['answer'][:200].strip()}...")
        print()
