from typing import List, Dict, Any

class HierarchicalManager:
    """Maps precise child chunks back to their encompassing parent document window."""
    def __init__(self, all_chunks: List[Dict[str, Any]]):
        self.chunk_index: Dict[str, Dict[str, Any]] = {
            str(c.get("id") or c.get("chunk_id", "")): c for c in all_chunks
        }
        self.parent_index: Dict[str, Dict[str, Any]] = {
            str(c.get("id") or c.get("chunk_id", "")): c 
            for c in all_chunks 
            if c.get("metadata", {}).get("chunk_type") == "parent"
        }

    def expand_to_parent_windows(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expands retrieved child chunks to full parent document windows while preserving lineage."""
        expanded_list = []
        seen_parents = set()

        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            parent_id = str(meta.get("parent_id", "")) if meta.get("parent_id") else None
            chunk_id = str(chunk.get("id") or chunk.get("chunk_id", ""))

            if parent_id and parent_id in self.parent_index:
                if parent_id not in seen_parents:
                    parent_chunk = self.parent_index[parent_id].copy()
                    parent_chunk["retrieval_trigger_child_id"] = chunk_id
                    parent_chunk["confidence_score"] = chunk.get("confidence_score", 1.0)
                    parent_chunk["rerank_score"] = chunk.get("rerank_score", None)
                    parent_chunk["rrf_score"] = chunk.get("rrf_score", None)
                    expanded_list.append(parent_chunk)
                    seen_parents.add(parent_id)
            else:
                expanded_list.append(chunk)

        return expanded_list