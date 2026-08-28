import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb

class DenseRetriever:
    """Production-grade Dense Vector Retriever backed by ChromaDB and BGE-Large-en-v1.5."""
    def __init__(
        self,
        collection_name: str = "day19_dense_corpus",
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

    def index_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> int:
        """Batch-embed and store document and figure chunks with sanitized metadata."""
        if not chunks:
            return 0
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            
            ids = [str(c.get("id") or c.get("chunk_id") or f"chunk_{i + idx}") for idx, c in enumerate(batch)]
            texts = [c.get("text") or c.get("content", "") for c in batch]
            
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
        top_k: int = 5,
        confidence_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Cosine similarity dense search with BGE instruction query prefix."""
        bge_query = f"Represent this sentence for searching relevant passages: {query}" if "bge" in self.model_name.lower() else query
        query_vec = self.embedder.encode([bge_query], normalize_embeddings=True).tolist()
        
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ranked = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            ids = results["ids"][0]

            for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
                conf = max(0.0, min(1.0, 1.0 - float(dist)))
                if confidence_threshold is not None and conf < confidence_threshold:
                    continue
                ranked.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "confidence_score": round(conf, 4),
                    "retriever_type": "dense_vector"
                })
        ranked.sort(key=lambda x: x["confidence_score"], reverse=True)
        return ranked