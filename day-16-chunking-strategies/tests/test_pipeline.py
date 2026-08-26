import os
import glob
import json
import pytest
from src.models import DocumentElement
from src.chunkers import (
    chunk_fixed_size,
    chunk_token_based,
    chunk_recursive,
    chunk_semantic,
    chunk_hierarchical
)

@pytest.fixture
def sample_document_elements():
    return [
        DocumentElement(
            content="Arthur Samuel coined the term Machine Learning in 1959. "
                    "Supervised learning maps inputs to outputs using labeled data. "
                    "Unsupervised learning groups unlabeled data into clusters.",
            metadata={
                "source": "SupportcoursesM-DLearning.pdf",
                "page_number": 9,
                "section_heading": "1.1 Introduction",
                "doc_type": "pdf_native"
            }
        ),
        DocumentElement(
            content="Free-tier API keys are limited to 60 requests per minute. "
                    "Professional keys allow 600 requests per minute. "
                    "Exceeding limits returns HTTP 429.",
            metadata={
                "source": "api_rate_limiting_policy.txt",
                "page_number": 1,
                "section_heading": "STANDARD RATE LIMITS",
                "doc_type": "txt"
            }
        )
    ]

def test_metadata_lineage_integrity(sample_document_elements):
    strategies = [
        chunk_fixed_size(sample_document_elements, 100, 10),
        chunk_token_based(sample_document_elements, 50, 10),
        chunk_recursive(sample_document_elements, 100, 10),
        chunk_semantic(sample_document_elements, 2),
        chunk_hierarchical(sample_document_elements, 200, 50)
    ]
    for chunks in strategies:
        assert len(chunks) > 0
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page_number" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "section_heading" in chunk.metadata
            assert chunk.chunk_id != ""
            assert len(chunk.content.strip()) > 0

# --- ADDED VERIFICATION TEST FOR ACTUAL OUTPUT JSON FILES ---
def test_actual_output_files_integrity():
    output_files = glob.glob("outputs/chunks_*.json")
    assert len(output_files) == 5, "Should have 5 strategy output files"
    
    for file_path in output_files:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            assert len(chunks) > 0
            for chunk in chunks:
                assert chunk["metadata"]["source"] != ""
                assert chunk["metadata"]["page_number"] >= 1
                assert chunk["metadata"]["section_heading"] != ""
                assert "chunk_index" in chunk["metadata"]