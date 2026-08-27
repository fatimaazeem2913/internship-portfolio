import os
import json
import pytest
import numpy as np
from src.embeddings import EmbeddingModelWrapper
from src.vector_stores import ChromaStoreManager, FAISSStoreManager

@pytest.fixture
def sample_data():
    return {
        "ids": ["doc_1", "doc_2", "doc_3"],
        "texts": [
            "Mean Squared Error loss formulation: MSE = 1/N sum((y - y_hat)^2).",
            "Tier 1 accounts throttled to 60 requests per minute with HTTP 429 status code.",
            "P99 latency SLA guarantees require responses under 250ms for 2MB payloads."
        ],
        "metadatas": [
            {"source": "SupportcoursesM-DLearning.pdf", "page": 18, "section": "Loss Formulations"},
            {"source": "api_rate_limiting_policy.txt", "page": 1, "section": "Rate Limits"},
            {"source": "product_spec.docx", "page": 3, "section": "SLA Metrics"}
        ]
    }

def test_embedding_wrapper():
    wrapper = EmbeddingModelWrapper("all-MiniLM-L6-v2")
    texts = ["Test document embedding", "Another sample sentence"]
    embeddings, elapsed = wrapper.embed_documents(texts)
    assert len(embeddings) == 2
    assert embeddings.shape[1] == 384
    assert elapsed >= 0.0

def test_faiss_vector_store(sample_data):
    dim = 384
    wrapper = EmbeddingModelWrapper("all-MiniLM-L6-v2")
    embeddings, _ = wrapper.embed_documents(sample_data["texts"])
    
    faiss_mgr = FAISSStoreManager(dimension=dim)
    ingest_time = faiss_mgr.add_documents(
        sample_data["ids"],
        sample_data["texts"],
        embeddings,
        sample_data["metadatas"]
    )
    assert faiss_mgr.count() == 3
    assert ingest_time >= 0.0

    q_vec, _ = wrapper.embed_query("What is the P99 SLA latency?")
    res, search_time = faiss_mgr.search(q_vec, top_k=1)
    assert len(res) == 1
    assert "metadata" in res[0]
    assert "source" in res[0]["metadata"]

def test_metadata_lineage_preservation():
    with open("data/chunks_hierarchical.json", "r") as f:
        chunks = json.load(f)
    assert len(chunks) >= 1000
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "metadata" in chunk
        meta = chunk["metadata"]
        assert all(k in meta for k in ["source", "page_number", "chunk_index", "section_heading"])
