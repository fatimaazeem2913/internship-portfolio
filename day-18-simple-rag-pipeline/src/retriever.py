import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb

class VectorRetriever:
    """High-performance vector retriever using BGE-Large with cosine similarity scoring & batch ingestion."""
    def __init__(
        self,
        collection_name: str = "rag_knowledge_base",
        model_name: str = "BAAI/bge-large-en-v1.5",
        persist_dir: str = "outputs/chroma_db"
    ):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> int:
        """Batch-embed and store document and figure chunks with metadata."""
        if not chunks:
            return 0

        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            
            # Robust key extraction for id and text/content
            ids = [str(c.get("id") or c.get("chunk_id") or f"chunk_{i + idx}") for idx, c in enumerate(batch)]
            texts = [c.get("text") or c.get("content", "") for c in batch]
            
            # Sanitize metadata for ChromaDB (primitive scalar types only)
            metadatas = []
            for c in batch:
                raw_meta = c.get("metadata", {})
                clean_meta = {}
                for k, v in raw_meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                metadatas.append(clean_meta)

            embeddings = self.embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
        return total

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Perform query embedding and cosine similarity top-k search with confidence scoring."""
        # BGE models benefit from instruction prefix for retrieval
        bge_query = f"Represent this sentence for searching relevant passages: {query}" if "bge" in self.model_name.lower() else query
        query_vec = self.embedder.encode([bge_query], normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ranked_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            ids = results["ids"][0]

            for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
                confidence_score = max(0.0, min(1.0, 1.0 - float(dist)))
                if score_threshold is not None and confidence_score < score_threshold:
                    continue
                ranked_results.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "confidence_score": round(confidence_score, 4)
                })

        ranked_results.sort(key=lambda x: x["confidence_score"], reverse=True)
        return ranked_results