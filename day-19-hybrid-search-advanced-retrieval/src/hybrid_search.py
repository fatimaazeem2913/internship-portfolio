from typing import List, Dict, Any
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever

def reciprocal_rank_fusion(
    rankings_list: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Computes Reciprocal Rank Fusion (RRF) across disparate ranking lists:
    RRF_Score(d) = sum_{m in models} 1 / (k + rank_m(d))
    """
    rrf_scores: Dict[str, float] = {}
    doc_registry: Dict[str, Dict[str, Any]] = {}
    origin_ranks: Dict[str, Dict[str, int]] = {}

    for ranking in rankings_list:
        for rank, item in enumerate(ranking, start=1):
            doc_id = str(item.get("id", ""))
            if not doc_id:
                continue

            if doc_id not in doc_registry:
                doc_registry[doc_id] = item.copy()
                origin_ranks[doc_id] = {}

            retriever_label = item.get("retriever_type", "unknown")
            origin_ranks[doc_id][retriever_label] = rank
            
            # Preserve individual scores
            if "confidence_score" in item:
                doc_registry[doc_id]["dense_confidence"] = item["confidence_score"]
            if "bm25_score" in item:
                doc_registry[doc_id]["bm25_score"] = item["bm25_score"]

            # Standard RRF additive score
            contribution = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

    # Sort merged results by RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

    fused_results = []
    for doc_id in sorted_doc_ids:
        base_item = doc_registry[doc_id].copy()
        base_item["rrf_score"] = round(rrf_scores[doc_id], 6)
        base_item["source_ranks"] = origin_ranks[doc_id]
        fused_results.append(base_item)

    return fused_results

class HybridSearcher:
    """Combines BM25 and Dense Vector search using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10, candidates_per_system: int = 20) -> List[Dict[str, Any]]:
        """Executes sparse + dense retrievals and fuses rankings."""
        sparse_hits = self.bm25.retrieve(query, top_k=candidates_per_system)
        dense_hits = self.dense.retrieve(query, top_k=candidates_per_system)
        
        fused = reciprocal_rank_fusion([sparse_hits, dense_hits], k=self.rrf_k)
        return fused[:top_k]