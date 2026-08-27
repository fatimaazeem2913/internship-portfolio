import pytest
import os
import json
from src.retriever import VectorRetriever
from src.prompt_builder import PromptConstructor
from src.llm_client import LLMClient
from src.pipeline import SimpleRAGPipeline

@pytest.fixture(scope="module")
def sample_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_chunks.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def rag_pipeline(sample_dataset, tmp_path_factory):
    temp_chroma = str(tmp_path_factory.mktemp("test_chroma"))
    retriever = VectorRetriever(collection_name="test_rag", persist_dir=temp_chroma)
    retriever.ingest_chunks(sample_dataset)
    llm = LLMClient(use_mock=True)
    return SimpleRAGPipeline(retriever=retriever, llm_client=llm)

def test_retrieval_confidence_and_ranking(rag_pipeline):
    """Verify retrieval returns ranked results with confidence scores."""
    results = rag_pipeline.retriever.retrieve("What is the formula for Mean Squared Error MSE?", top_k=2)
    assert len(results) > 0
    top_hit = results[0]
    assert "confidence_score" in top_hit
    assert top_hit["confidence_score"] > 0.50
    assert "metadata" in top_hit
    assert top_hit["metadata"]["source"] == "ml_fundamentals_guide.pdf"

def test_prompt_construction_and_citation_contract():
    """Verify PromptConstructor injects metadata tags and citation rules."""
    mock_chunks = [{
        "id": "c1",
        "text": "Sample text content",
        "metadata": {"source": "manual.pdf", "page_number": 5, "section_heading": "Overview"},
        "confidence_score": 0.92
    }]
    prompt = PromptConstructor.build("What is sample?", mock_chunks)
    assert "manual.pdf" in prompt["context_str"]
    assert "Page 5" in prompt["context_str"]
    assert "cite the exact source document" in prompt["system_instruction"].lower()

def test_e2e_4_question_types(rag_pipeline):
    """Verify end-to-end pipeline against all 4 benchmark question archetypes."""
    eval_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for q_item in questions:
        res = rag_pipeline.run(q_item["question"], top_k=2)
        q_type = q_item["type"]
        answer = res["answer"]
        
        if q_type == "factual_lookup":
            assert "MSE" in answer or "Mean Squared Error" in answer
            assert "[Source:" in answer
        elif q_type == "inference_required":
            assert "384-dimension" in answer or "latency" in answer
            assert "[Source:" in answer
        elif q_type == "cross_document":
            assert "[Source:" in answer
        elif q_type == "out_of_domain":
            assert "does not contain sufficient information" in answer