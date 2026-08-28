from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Second-stage cross-encoder re-ranking model for fine-grained semantic relevance."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Computes joint query-passage cross-attention logits to re-score top candidate chunks."""
        if not candidates:
            return []

        pairs = [[query, c.get("text") or c.get("content", "")] for c in candidates]
        scores = self.reranker.predict(pairs)

        reranked = []
        for c, score in zip(candidates, scores):
            item = c.copy()
            item["rerank_score"] = round(float(score), 4)
            reranked.append(item)

        # Sort descending by cross-encoder logit score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]