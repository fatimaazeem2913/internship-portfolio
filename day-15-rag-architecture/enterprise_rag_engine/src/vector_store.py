"""
vector_store.py
----------------
Embedding -> Vector Store.

Wraps ChromaDB (persistent, on-disk) as the vector store.

Real bug already documented for this project: ChromaDB's DEFAULT embedding
function auto-downloads an ONNX model over the network on first use, which
fails in network-restricted environments with a cryptic SHA256 error. Fix
applied throughout this module: `embedding_function` is NEVER set on the
collection. Embeddings are always computed explicitly (embedding.py) and
passed in via the `embeddings=` argument on add() and query(), so Chroma
never tries to embed anything itself.
"""

from __future__ import annotations

import chromadb

from chunking import Chunk
from embedding import embed_texts, embed_query


class VectorStore:
    def __init__(self, persist_dir: str = "data/vector_store", collection_name: str = "rag_chunks"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # embedding_function=None is the key line that avoids the ONNX
        # auto-download bug -- we supply embeddings= explicitly instead.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        """Drop and recreate the collection (useful for repeatable test runs)."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[{"doc_id": c.doc_id, "source": c.source, **{
                k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))
            }} for c in chunks],
        )

    def query(self, query_text: str, top_k: int = 4) -> list[dict]:
        query_vector = embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    from ingestion import load_corpus
    from chunking import chunk_documents

    docs = load_corpus("data/corpus")
    chunks = chunk_documents(docs)

    store = VectorStore(persist_dir="data/vector_store_test")
    store.reset()
    store.add_chunks(chunks)
    print(f"Indexed {store.count()} chunks into ChromaDB (no ONNX auto-download used).")

    results = store.query("How long do I have to return a defective item?", top_k=3)
    print("\nTop 3 results for: 'How long do I have to return a defective item?'\n")
    for r in results:
        print(f"[{r['chunk_id']}] distance={r['distance']:.4f}")
        print(r["text"][:150].replace("\n", " ") + "...\n")
