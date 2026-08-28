import pytest
import os
import json
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_search import HybridSearcher, reciprocal_rank_fusion
from src.query_rewriter import QueryRewriter
from src.reranker import CrossEncoderReranker
from src.hierarchical_manager import HierarchicalManager
from src.pipeline_advanced import AdvancedRAGPipeline

@pytest.fixture(scope="module")
def corpus():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_corpus.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def pipeline(corpus, tmp_path_factory):
    # Isolated test pipeline
    p = AdvancedRAGPipeline(corpus_chunks=corpus, use_mock=True)
    return p

def test_bm25_exact_keyword_retrieval(pipeline):
    """Test 1: BM25 excels at exact keyword and error-code lookups."""
    results = pipeline.bm25.retrieve("ERR_CONN_TIMEOUT_403 port 8000", top_k=2)
    assert len(results) > 0
    assert "ERR_CONN_TIMEOUT_403" in results[0]["text"]
    assert results[0]["bm25_score"] > 0

def test_reciprocal_rank_fusion_math():
    """Test 2: Verify RRF scoring math and rank aggregation."""
    list1 = [{"id": "doc_A", "retriever_type": "bm25"}, {"id": "doc_B", "retriever_type": "bm25"}]
    list2 = [{"id": "doc_B", "retriever_type": "dense"}, {"id": "doc_A", "retriever_type": "dense"}]
    
    # k=60: doc_A has ranks 1 & 2 -> 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # doc_B has ranks 2 & 1 -> exactly equal
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 2
    assert "rrf_score" in fused[0]
    assert pytest.approx(fused[0]["rrf_score"], rel=1e-3) == 0.032522

def test_query_rewriter_correction(pipeline):
    """Test 3: Query rewriter corrects severe typos and expands acronyms."""
    typo_q = "what is slinear reggression hypotthesis formula"
    rewritten = pipeline.rewriter.rewrite(typo_q)
    assert "linear regression" in rewritten.lower()
    
    acronym_q = "What is CE loss formulation?"
    rewritten_acronym = pipeline.rewriter.rewrite(acronym_q)
    assert "cross entropy" in rewritten_acronym.lower()

def test_cross_encoder_reranker_scoring(pipeline):
    """Test 4: Cross-Encoder produces continuous relevance scores and ranks correctly."""
    candidates = [
        {"id": "irr", "text": "Pizza baking instructions in woodfired ovens."},
        {"id": "rel", "text": "BM25Okapi scoring hyperparameter k1=1.5 governs term frequency saturation."}
    ]
    reranked = pipeline.reranker.rerank("BM25 parameters k1 and b", candidates, top_k=2)
    assert reranked[0]["id"] == "rel"
    assert "rerank_score" in reranked[0]
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]

def test_hierarchical_parent_window_expansion(corpus):
    """Test 5: Hierarchical manager maps child chunk hit to encompassing parent document."""
    manager = HierarchicalManager(corpus)
    child_chunk = [c for c in corpus if c["id"] == "c1_1_mse"][0]
    expanded = manager.expand_to_parent_windows([child_chunk])
    
    assert len(expanded) == 1
    assert expanded[0]["id"] == "p1_parent"
    assert "Supervised learning algorithms" in expanded[0]["text"]

def test_e2e_advanced_pipeline_comparison(pipeline):
    """Test 6: End-to-end execution of all 4 retrieval paradigms on benchmark questions."""
    q = "What is the mathematical loss formula for Mean Squared Error MSE?"
    
    res_dense = pipeline.run(q, mode="simple_dense")
    res_bm25 = pipeline.run(q, mode="bm25_sparse")
    res_hybrid = pipeline.run(q, mode="hybrid_rrf")
    res_rerank = pipeline.run(q, mode="hybrid_rerank")

    for res in [res_dense, res_bm25, res_hybrid, res_rerank]:
        assert "MSE" in res["answer"] or "Mean Squared Error" in res["answer"]
        assert len(res["citations"]) > 0
