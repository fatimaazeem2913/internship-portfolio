import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """Production-grade Sparse Lexical Retriever based on BM25Okapi."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Alphanumeric tokenization with lowercase normalization."""
        return re.findall(r"\w+", text.lower())

    def index_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """Indexes raw document chunks for lexical scoring."""
        self.corpus_chunks = chunks
        self.tokenized_corpus = [self.tokenize(c.get("text", c.get("content", ""))) for c in chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        return len(self.corpus_chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Scores and ranks corpus chunks using BM25Okapi."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            chunk = self.corpus_chunks[idx]
            results.append({
                "id": str(chunk.get("id") or chunk.get("chunk_id") or f"chunk_{idx}"),
                "text": chunk.get("text", chunk.get("content", "")),
                "metadata": chunk.get("metadata", {}),
                "bm25_score": round(score, 4),
                "retriever_type": "bm25_sparse"
            })
        return results