"""
test_pipeline.py
-----------------
Real, executable tests for every stage of the RAG pipeline. Run offline
(USE_MOCK_LLM=true is set in conftest-style fixture below) so the suite
runs with zero API key and zero dependency on huggingface.co, matching
this project's established free/offline-first testing pattern.

Run with:  venv/bin/python -m pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("USE_MOCK_LLM", "true")

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from ingestion import load_corpus, Document          # noqa: E402
from chunking import chunk_document, chunk_documents  # noqa: E402
from embedding import embed_texts, embed_query, reset_model  # noqa: E402
from vector_store import VectorStore                  # noqa: E402
from retrieval import Retriever                        # noqa: E402
from llm import generate_answer, build_augmented_prompt  # noqa: E402
from pipeline import RAGPipeline                        # noqa: E402


TEST_VECTOR_STORE_DIR = str(PROJECT_ROOT / "data" / "vector_store_test")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def test_load_corpus_finds_both_files():
    docs = load_corpus(str(PROJECT_ROOT / "data" / "corpus"))
    doc_ids = {d.doc_id for d in docs}
    assert "refund_policy" in doc_ids
    assert "shipping_policy" in doc_ids


def test_load_corpus_extracts_real_text_not_empty():
    docs = load_corpus(str(PROJECT_ROOT / "data" / "corpus"))
    for d in docs:
        assert len(d.text) > 100, f"{d.doc_id} extracted suspiciously little text"


def test_pdf_extraction_actually_worked():
    """Regression test for the ingestion.py pdfplumber/pypdf fallback path --
    confirms the PDF corpus file yields real extracted text, not silence."""
    docs = load_corpus(str(PROJECT_ROOT / "data" / "corpus"))
    pdf_doc = next(d for d in docs if d.doc_id == "shipping_policy")
    assert "SECTION 1" in pdf_doc.text.upper()
    assert "OVERNIGHT" in pdf_doc.text.upper()


def test_missing_corpus_dir_raises():
    with pytest.raises(FileNotFoundError):
        load_corpus("data/does_not_exist")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunking_produces_chunks():
    doc = Document(doc_id="test", source="test.txt", text="Hello world. " * 200)
    chunks = chunk_document(doc, chunk_size=50, overlap=10)
    assert len(chunks) > 1


def test_chunks_respect_approximate_size():
    doc = Document(doc_id="test", source="test.txt", text="word " * 500)
    chunks = chunk_document(doc, chunk_size=50, overlap=10)
    for c in chunks:
        word_count = len(c.text.split())
        # allow slack since paragraph-boundary splitting means chunks won't
        # be exactly chunk_size
        assert word_count <= 60, f"chunk exceeded target size: {word_count} words"


def test_chunk_ids_are_unique():
    docs = load_corpus(str(PROJECT_ROOT / "data" / "corpus"))
    chunks = chunk_documents(docs)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_id found"


def test_every_chunk_traces_back_to_source_doc():
    docs = load_corpus(str(PROJECT_ROOT / "data" / "corpus"))
    chunks = chunk_documents(docs)
    doc_ids = {d.doc_id for d in docs}
    for c in chunks:
        assert c.doc_id in doc_ids


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def test_embed_texts_returns_one_vector_per_input():
    vectors = embed_texts(["hello", "world", "foo"])
    assert len(vectors) == 3


def test_embed_vectors_are_nonzero():
    # Reset first: the fallback TF-IDF vectorizer's vocabulary is fixed at
    # first fit (see embedding.py docstring). Without resetting, whichever
    # test happens to run first "claims" the vocabulary and later tests'
    # words can come back all-zero (out-of-vocabulary) -- a real, order-
    # dependent bug this project hit and fixed during development.
    reset_model()
    vectors = embed_texts(["refund policy text here"])
    assert any(abs(x) > 0 for x in vectors[0])


def test_embed_query_matches_embed_texts_dimension():
    reset_model()
    doc_vec = embed_texts(["some document chunk"])[0]
    query_vec = embed_query("some query")
    assert len(doc_vec) == len(query_vec)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def test_vector_store_round_trip():
    docs = load_corpus(str(PROJECT_ROOT / "data" / "corpus"))
    chunks = chunk_documents(docs)
    store = VectorStore(persist_dir=TEST_VECTOR_STORE_DIR, collection_name="test_round_trip")
    store.reset()
    store.add_chunks(chunks)
    assert store.count() == len(chunks)


def test_vector_store_query_returns_results():
    docs = load_corpus(str(PROJECT_ROOT / "data" / "corpus"))
    chunks = chunk_documents(docs)
    store = VectorStore(persist_dir=TEST_VECTOR_STORE_DIR, collection_name="test_query")
    store.reset()
    store.add_chunks(chunks)
    results = store.query("refund", top_k=3)
    assert len(results) == 3
    assert all("chunk_id" in r for r in results)


def test_no_onnx_default_embedder_used():
    """Regression test for the documented ChromaDB default-embedder bug --
    confirms embedding_function is explicitly disabled on the collection."""
    store = VectorStore(persist_dir=TEST_VECTOR_STORE_DIR, collection_name="test_no_onnx")
    assert store.collection._embedding_function is None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built_retriever():
    docs = load_corpus(str(PROJECT_ROOT / "data" / "corpus"))
    chunks = chunk_documents(docs)
    store = VectorStore(persist_dir=TEST_VECTOR_STORE_DIR, collection_name="test_retrieval")
    store.reset()
    store.add_chunks(chunks)
    return Retriever(chunks, store)


def test_bm25_exact_term_match_wins(built_retriever):
    """BM25 should surface the exact section on a query built from its
    literal heading words."""
    results = built_retriever.bm25_retrieve("SECTION 5 late refunds", top_k=1)
    assert "SECTION 5" in results[0]["text"].upper()


def test_dense_retrieve_returns_top_k(built_retriever):
    results = built_retriever.dense_retrieve("shipping timelines", top_k=2)
    assert len(results) == 2


def test_hybrid_retrieve_returns_top_k(built_retriever):
    results = built_retriever.hybrid_retrieve("international shipping customs", top_k=3)
    assert len(results) == 3
    assert all("rrf_score" in r for r in results)


# ---------------------------------------------------------------------------
# LLM / prompt construction
# ---------------------------------------------------------------------------

def test_augmented_prompt_includes_context_and_question():
    fake_hits = [{"text": "Refunds take 5-7 days.", "metadata": {"source": "refund_policy.txt"}}]
    prompt = build_augmented_prompt("How long do refunds take?", fake_hits)
    assert "Refunds take 5-7 days." in prompt
    assert "How long do refunds take?" in prompt


def test_mock_llm_generates_answer_without_network_or_key():
    fake_hits = [{"text": "Refunds take 5-7 days.", "metadata": {"source": "refund_policy.txt"}}]
    result = generate_answer("How long do refunds take?", fake_hits)
    assert result["backend"] == "mock"
    assert "5-7 days" in result["answer"]
    assert result["sources"] == ["refund_policy.txt"]


def test_generate_answer_with_no_chunks_says_so():
    result = generate_answer("Unanswerable question", [])
    assert "don't have enough information" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Full pipeline (integration)
# ---------------------------------------------------------------------------

def test_full_pipeline_end_to_end():
    pipeline = RAGPipeline(
        persist_dir=str(PROJECT_ROOT / "data" / "vector_store_test"),
        retrieval_mode="hybrid",
        top_k=3,
    )
    n_chunks = pipeline.build_index()
    assert n_chunks > 0

    result = pipeline.query("How long do I have to report a damaged item?")
    assert result["backend"] == "mock"
    assert "refund_policy.txt" in result["sources"]
    assert len(result["retrieved_chunks"]) == 3


def test_pipeline_raises_if_queried_before_index_built():
    pipeline = RAGPipeline(persist_dir=str(PROJECT_ROOT / "data" / "vector_store_test"))
    with pytest.raises(RuntimeError):
        pipeline.query("anything")
