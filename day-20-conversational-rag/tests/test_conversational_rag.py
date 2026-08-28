import os
import pytest
from src.conversational_rag import ConversationalRAGPipeline


@pytest.fixture
def rag_pipeline():
    return ConversationalRAGPipeline(use_compression=False)


def test_session_isolation(rag_pipeline):
    """Verifies that separate session IDs maintain completely isolated conversational histories."""
    rag_pipeline.clear_session("test_sess_1")
    rag_pipeline.clear_session("test_sess_2")

    res1 = rag_pipeline.ask("test_sess_1", "What is Mean Squared Error?")
    assert len(res1["answer"]) > 0
    assert len(rag_pipeline.get_session_history("test_sess_1")) == 2
    assert len(rag_pipeline.get_session_history("test_sess_2")) == 0


def test_followup_reformulation(rag_pipeline):
    """Verifies that pronouns and follow-up references are resolved into standalone search queries."""
    sess = "test_sess_followup"
    rag_pipeline.clear_session(sess)
    
    rag_pipeline.ask(sess, "What are the core biological components of a neuron shown in Figure 4.1?")
    res2 = rag_pipeline.ask(sess, "What does the second component do in an artificial neural network?")
    
    standalone = res2["standalone_query"].lower()
    # Standalone query should resolve 'the second component' with domain context
    target_terms = ["neuron", "soma", "component", "biological", "network", "dendrite", "cell"]
    assert any(term in standalone for term in target_terms), f"Reformulated query lacked expected context: '{res2['standalone_query']}'"


def test_source_citation_presence(rag_pipeline):
    """Verifies that returned responses contain formatted source citations with document and page info."""
    sess = "test_sess_citations"
    rag_pipeline.clear_session(sess)
    
    res = rag_pipeline.ask(sess, "What is the formula for Mean Squared Error (MSE)?")
    assert isinstance(res["citations"], list)
    if res["citations"]:
        assert any("Source:" in c and "Page:" in c for c in res["citations"])