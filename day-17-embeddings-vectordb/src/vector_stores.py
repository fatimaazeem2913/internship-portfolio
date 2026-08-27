import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple

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
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception:
            self.client = None
            self.collection = None

    def add_documents(self, ids: List[str], documents: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> float:
        start_time = time.time()
        if self.collection is not None:
            # Ensure embeddings is a list
            if isinstance(embeddings, np.ndarray):
                emb_list = embeddings.tolist()
            else:
                emb_list = list(embeddings)
                
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=emb_list,
                metadatas=metadatas
            )
        elapsed = time.time() - start_time
        return elapsed

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> Tuple[List[Dict[str, Any]], float]:
        start_time = time.time()
        results = []
        if self.collection is not None and self.count() > 0:
            query_arr = np.array(query_embedding, dtype=np.float32)
            if query_arr.ndim == 1:
                query_list = [query_arr.tolist()]
            else:
                query_list = query_arr.tolist()

            k = min(top_k, self.count())
            res = self.collection.query(
                query_embeddings=query_list,
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            if res and "ids" in res and len(res["ids"]) > 0:
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
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass


class FAISSStoreManager:
    """Manages FAISS IndexFlatIP (Cosine similarity) with document and metadata indexing."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.docs = []
        self.metadatas = []
        self.ids = []
        self.vectors = []
        self._init_faiss()

    def _init_faiss(self):
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
        except Exception:
            self.index = None

    def add_documents(self, ids: List[str], documents: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> float:
        start_time = time.time()
        emb_arr = np.array(embeddings, dtype=np.float32)
        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(1, -1)

        # Normalize vectors for Cosine IP
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        norm_embeddings = emb_arr / norms

        if self.index is not None:
            self.index.add(norm_embeddings.astype(np.float32))

        self.ids.extend(ids)
        self.docs.extend(documents)
        self.metadatas.extend(metadatas)

        if len(self.vectors) == 0:
            self.vectors = norm_embeddings
        else:
            self.vectors = np.vstack([self.vectors, norm_embeddings])

        elapsed = time.time() - start_time
        return elapsed

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> Tuple[List[Dict[str, Any]], float]:
        start_time = time.time()
        results = []
        total_docs = self.count()

        if total_docs == 0:
            return results, time.time() - start_time

        # Ensure query is 2D with shape (1, dimension)
        q_arr = np.array(query_embedding, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)

        # Normalize query vector
        q_norm_val = np.linalg.norm(q_arr)
        if q_norm_val == 0:
            q_norm_val = 1e-9
        query_norm = q_arr / q_norm_val

        k = min(top_k, total_docs)

        if self.index is not None and self.index.ntotal > 0:
            scores, indices = self.index.search(query_norm.astype(np.float32), k)
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.docs):
                    results.append({
                        "id": self.ids[idx],
                        "document": self.docs[idx],
                        "metadata": self.metadatas[idx],
                        "score": float(score)
                    })
        else:
            # Fallback numpy cosine similarity
            if len(self.vectors) > 0:
                sims = np.dot(self.vectors, query_norm.T).flatten()
                top_indices = np.argsort(-sims)[:k]
                for idx in top_indices:
                    results.append({
                        "id": self.ids[idx],
                        "document": self.docs[idx],
                        "metadata": self.metadatas[idx],
                        "score": float(sims[idx])
                    })

        elapsed = time.time() - start_time
        return results, elapsed

    def count(self) -> int:
        if self.index is not None:
            return self.index.ntotal
        return len(self.docs)