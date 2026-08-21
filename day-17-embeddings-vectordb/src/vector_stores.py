import os
import json
import time
import numpy as np
from typing import List, Dict, Any

class ChromaStoreManager:
    """Manages ChromaDB collections with full metadata filtering and persistence."""
    def __init__(self, collection_name: str = "rag_day17", persist_dir: str = "./outputs/chroma_db"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception:
            self.client = None
            self.collection = None

    def add_documents(self, ids: List[str], documents: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> float:
        start_time = time.time()
        if self.collection is not None:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
        elapsed = time.time() - start_time
        return elapsed

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> (List[Dict[str, Any]], float):
        start_time = time.time()
        results = []
        if self.collection is not None:
            res = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            for i in range(len(res["ids"][0])):
                results.append({
                    "id": res["ids"][0][i],
                    "document": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "distance": res["distances"][0][i]
                })
        elapsed = time.time() - start_time
        return results, elapsed

    def count(self) -> int:
        return self.collection.count() if self.collection else 0

    def delete_collection(self):
        if self.client:
            self.client.delete_collection(name=self.collection_name)


class FAISSStoreManager:
    """Manages FAISS IndexFlatIP (Cosine similarity) with document and metadata indexing."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.docs = []
        self.metadatas = []
        self.ids = []
        self._init_faiss()

    def _init_faiss(self):
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
        except Exception:
            self.index = None

    def add_documents(self, ids: List[str], documents: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> float:
        start_time = time.time()
        # Ensure vectors are normalized for Cosine IP
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_embeddings = embeddings / (norms + 1e-9)
        
        if self.index is not None:
            self.index.add(norm_embeddings.astype(np.float32))
        
        self.ids.extend(ids)
        self.docs.extend(documents)
        self.metadatas.extend(metadatas)
        elapsed = time.time() - start_time
        return elapsed

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> (List[Dict[str, Any]], float):
        start_time = time.time()
        results = []
        if self.index is not None and self.index.ntotal > 0:
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            scores, indices = self.index.search(query_norm.astype(np.float32), top_k)
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.docs):
                    results.append({
                        "id": self.ids[idx],
                        "document": self.docs[idx],
                        "metadata": self.metadatas[idx],
                        "score": float(score)
                    })
        elapsed = time.time() - start_time
        return results, elapsed

    def count(self) -> int:
        return self.index.ntotal if self.index else len(self.docs)
