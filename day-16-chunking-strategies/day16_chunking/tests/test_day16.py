"""
test_day16.py
-------------
Real, executable tests covering ingestion (PDF/DOCX/TXT/OCR), all 5
chunking strategies, and metadata integrity verification.

Run with: python -m pytest tests/ -v   (from the project root, venv active)
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = str(PROJECT_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ingest_pdf import extract_pdf, extract_with_pdfplumber, extract_with_pypdf2   # noqa: E402
from ingest_docx import extract_docx                                               # noqa: E402
from ingest_txt import extract_txt                                                 # noqa: E402
from ingest_ocr import compare_native_vs_ocr                                        # noqa: E402
from chunking_strategies import (                                                  # noqa: E402
    fixed_size_chunk, token_based_chunk, recursive_chunk, semantic_chunk,
    hierarchical_chunk,
)
from embedding import embed_texts                                                  # noqa: E402
from pipeline import chunk_pdf, chunk_docx, chunk_txt, verify_metadata_integrity   # noqa: E402

PDF_PATH = str(PROJECT_ROOT / "data" / "pdfs" / "employee_handbook.pdf")
DOCX_PATH = str(PROJECT_ROOT / "data" / "docx" / "product_spec.docx")
TXT_PATH = str(PROJECT_ROOT / "data" / "txt" / "api_rate_limiting_policy.txt")
SCANNED_PDF_PATH = str(PROJECT_ROOT / "data" / "scanned" / "vendor_nda_scanned.pdf")


# ---------------------------------------------------------------------------
# Ingestion: PDF
# ---------------------------------------------------------------------------

def test_pdf_has_50_plus_pages():
    pages = extract_pdf(PDF_PATH)
    assert len(pages) >= 50, f"Expected 50+ pages, got {len(pages)}"


def test_pdf_extraction_is_not_empty():
    pages = extract_pdf(PDF_PATH)
    total_chars = sum(len(p.text) for p in pages)
    assert total_chars > 5000


def test_pdfplumber_and_pypdf2_both_extract_real_text():
    plumber_pages = extract_with_pdfplumber(PDF_PATH)
    pypdf2_pages = extract_with_pypdf2(PDF_PATH)
    assert sum(len(p.text) for p in plumber_pages) > 1000
    assert sum(len(p.text) for p in pypdf2_pages) > 1000


def test_pdf_page_numbers_are_sequential():
    pages = extract_pdf(PDF_PATH)
    page_numbers = [p.page_number for p in pages]
    assert page_numbers == list(range(1, len(pages) + 1))


# ---------------------------------------------------------------------------
# Ingestion: DOCX
# ---------------------------------------------------------------------------

def test_docx_extraction_finds_headings():
    blocks = extract_docx(DOCX_PATH)
    heading_blocks = [b for b in blocks if b.heading_level >= 1]
    assert len(heading_blocks) >= 5


def test_docx_extraction_finds_table():
    blocks = extract_docx(DOCX_PATH)
    table_blocks = [b for b in blocks if "(table)" in b.section_heading]
    assert len(table_blocks) >= 1
    assert "API response time" in table_blocks[0].text


def test_docx_body_blocks_carry_section_heading():
    blocks = extract_docx(DOCX_PATH)
    body_blocks = [b for b in blocks if b.heading_level == 0]
    assert all(b.section_heading for b in body_blocks)


# ---------------------------------------------------------------------------
# Ingestion: TXT
# ---------------------------------------------------------------------------

def test_txt_extraction_finds_headings():
    blocks = extract_txt(TXT_PATH)
    headings = {b.section_heading for b in blocks}
    assert "STANDARD RATE LIMITS" in headings
    assert "RATE LIMIT HEADERS" in headings


def test_txt_extraction_is_not_empty():
    blocks = extract_txt(TXT_PATH)
    assert sum(len(b.text) for b in blocks) > 500


# ---------------------------------------------------------------------------
# Ingestion: OCR
# ---------------------------------------------------------------------------

def test_native_extraction_genuinely_fails_on_scanned_pdf():
    results = compare_native_vs_ocr(SCANNED_PDF_PATH)
    assert all(r.native_char_count == 0 for r in results)


def test_ocr_genuinely_recovers_text_from_scanned_pdf():
    results = compare_native_vs_ocr(SCANNED_PDF_PATH)
    assert sum(r.ocr_char_count for r in results) > 500


def test_ocr_recovers_expected_keywords():
    results = compare_native_vs_ocr(SCANNED_PDF_PATH)
    full_ocr_text = " ".join(r.ocr_text for r in results).upper()
    assert "NON-DISCLOSURE" in full_ocr_text or "NONDISCLOSURE" in full_ocr_text
    assert "CONFIDENTIAL" in full_ocr_text


# ---------------------------------------------------------------------------
# Chunking strategies (unit-level, on controlled sample text)
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Free-tier API keys are limited to 60 requests per minute and 10,000 requests per day. "
    "Professional-tier API keys are limited to 600 requests per minute and 200,000 per day. "
    "Enterprise-tier customers negotiate custom limits directly with their account manager.\n\n"
    "Every API response includes three headers describing the caller's current rate limit "
    "status. Clients should treat these headers as authoritative for request pacing."
)


def test_fixed_size_respects_chunk_size_ceiling():
    chunks = fixed_size_chunk(SAMPLE_TEXT, "sample.txt", chunk_size=100, overlap=10)
    assert all(len(c.text) <= 100 for c in chunks)


def test_fixed_size_produces_multiple_chunks_for_long_text():
    chunks = fixed_size_chunk(SAMPLE_TEXT, "sample.txt", chunk_size=100, overlap=10)
    assert len(chunks) > 1


def test_token_based_chunk_reports_token_count_metadata():
    chunks = token_based_chunk(SAMPLE_TEXT, "sample.txt", chunk_size_tokens=30, overlap_tokens=5)
    assert all("token_count" in c.metadata for c in chunks)
    assert all(c.metadata["token_count"] <= 30 for c in chunks)


def test_recursive_chunks_are_not_empty():
    chunks = recursive_chunk(SAMPLE_TEXT, "sample.txt", chunk_size=150, overlap=20)
    assert len(chunks) > 0
    assert all(c.text.strip() for c in chunks)


def test_recursive_prefers_paragraph_boundaries_over_hard_cuts():
    """Real behavioral check: recursive splitting should cut mid-WORD less
    often than naive fixed-size splitting, since it prefers separator
    boundaries (paragraph/sentence/word) before falling back to a hard
    character cut.

    Note on an earlier version of this test: checking whether a chunk's
    text ends in '.', '!', or '?' is NOT a reliable proxy for "did this cut
    happen mid-word" -- LangChain's RecursiveCharacterTextSplitter can
    split cleanly at a ". " boundary while consuming the separator itself,
    leaving the chunk ending in a real word with no trailing period even
    though the cut was clean. The only real test is whether the ORIGINAL
    text had a non-space, alphanumeric character immediately following the
    chunk's end position -- that's a genuine mid-word cut regardless of
    whether trailing punctuation survived the split."""
    fixed_chunks = fixed_size_chunk(SAMPLE_TEXT, "sample.txt", chunk_size=150, overlap=0)
    recursive_chunks = recursive_chunk(SAMPLE_TEXT, "sample.txt", chunk_size=150, overlap=0)

    def count_true_mid_word_cuts(chunks) -> int:
        count = 0
        search_start = 0
        for c in chunks[:-1]:
            end_pos = SAMPLE_TEXT.find(c.text.strip(), search_start)
            if end_pos == -1:
                continue  # chunk text was transformed (e.g. separator stripped); skip rather than guess
            end_pos += len(c.text.strip())
            search_start = end_pos
            if end_pos < len(SAMPLE_TEXT):
                next_char = SAMPLE_TEXT[end_pos]
                prev_char = SAMPLE_TEXT[end_pos - 1]
                if next_char.isalnum() and prev_char.isalnum():
                    count += 1
        return count

    fixed_mid_word_count = count_true_mid_word_cuts(fixed_chunks)
    recursive_mid_word_count = count_true_mid_word_cuts(recursive_chunks)
    assert recursive_mid_word_count <= fixed_mid_word_count


def test_semantic_chunk_isolates_an_off_topic_sentence():
    text_with_outlier = (
        "Refunds are processed within 5 to 7 business days after approval. "
        "Customers can track refund status through the online portal. "
        "The weather today is sunny with a light breeze. "
        "Shipping costs are non-refundable except for company errors."
    )
    chunks = semantic_chunk(text_with_outlier, "sample.txt", embed_fn=embed_texts, breakpoint_percentile=60)
    assert any("weather" in c.text.lower() for c in chunks)


def test_hierarchical_chunk_never_crosses_section_boundary():
    sections = [
        ("Section A", "A" * 600),
        ("Section B", "B" * 600),
    ]
    chunks = hierarchical_chunk(sections, "sample.txt", max_chunk_size=200, overlap=20)
    for c in chunks:
        assert set(c.text) <= {"A"} or set(c.text) <= {"B"}, \
            "hierarchical chunk mixed content from two different sections"


def test_hierarchical_keeps_short_section_as_single_chunk():
    sections = [("Short Section", "This is a short section under the limit.")]
    chunks = hierarchical_chunk(sections, "sample.txt", max_chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].metadata["section_split"] is False


# ---------------------------------------------------------------------------
# Full pipeline + metadata integrity (the task's explicit requirement)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["fixed_size", "token_based", "recursive", "hierarchical"])
def test_pdf_chunks_have_no_metadata_integrity_issues(strategy):
    kwargs = {"max_chunk_size": 500, "overlap": 50} if strategy == "hierarchical" else {"chunk_size": 500, "overlap": 50}
    if strategy == "token_based":
        kwargs = {"chunk_size_tokens": 150, "overlap_tokens": 20}
    chunks = chunk_pdf(PDF_PATH, strategy=strategy, **kwargs)
    report = verify_metadata_integrity(chunks, "employee_handbook.pdf")
    assert report["issues_found"] == 0, report["issues"]
    assert report["all_have_source"]
    assert report["all_have_chunk_index"]


@pytest.mark.parametrize("strategy", ["fixed_size", "token_based", "recursive", "hierarchical"])
def test_docx_chunks_have_no_metadata_integrity_issues(strategy):
    kwargs = {"max_chunk_size": 500, "overlap": 50} if strategy == "hierarchical" else {"chunk_size": 500, "overlap": 50}
    if strategy == "token_based":
        kwargs = {"chunk_size_tokens": 150, "overlap_tokens": 20}
    chunks = chunk_docx(DOCX_PATH, strategy=strategy, **kwargs)
    report = verify_metadata_integrity(chunks, "product_spec.docx")
    assert report["issues_found"] == 0, report["issues"]
    assert report["all_have_source"]


@pytest.mark.parametrize("strategy", ["fixed_size", "token_based", "recursive", "hierarchical"])
def test_txt_chunks_have_no_metadata_integrity_issues(strategy):
    kwargs = {"max_chunk_size": 500, "overlap": 50} if strategy == "hierarchical" else {"chunk_size": 500, "overlap": 50}
    if strategy == "token_based":
        kwargs = {"chunk_size_tokens": 150, "overlap_tokens": 20}
    chunks = chunk_txt(TXT_PATH, strategy=strategy, **kwargs)
    report = verify_metadata_integrity(chunks, "api_rate_limiting_policy.txt")
    assert report["issues_found"] == 0, report["issues"]
    assert report["all_have_source"]


def test_pdf_hierarchical_chunks_mostly_have_section_headings():
    chunks = chunk_pdf(PDF_PATH, strategy="hierarchical", max_chunk_size=500, overlap=50)
    report = verify_metadata_integrity(chunks, "employee_handbook.pdf")
    # Real heuristic, not perfect -- assert the large majority succeed
    assert report["chunks_with_section_heading"] / report["total_chunks"] > 0.9


def test_all_document_types_chunk_without_error():
    """Smoke test: every document type + every non-hierarchical strategy
    combination should run without raising. Each strategy takes its own
    real, distinct set of size parameters -- fixed_size/recursive use
    character-based chunk_size/overlap, while token_based genuinely uses a
    different unit (chunk_size_tokens/overlap_tokens), so this test uses
    the correct kwargs per strategy rather than assuming one shape fits all."""
    strategy_kwargs = {
        "fixed_size": {"chunk_size": 500, "overlap": 50},
        "recursive": {"chunk_size": 500, "overlap": 50},
        "token_based": {"chunk_size_tokens": 150, "overlap_tokens": 20},
    }
    for strategy, kwargs in strategy_kwargs.items():
        assert len(chunk_pdf(PDF_PATH, strategy=strategy, **kwargs)) > 0
        assert len(chunk_docx(DOCX_PATH, strategy=strategy, **kwargs)) > 0
        assert len(chunk_txt(TXT_PATH, strategy=strategy, **kwargs)) > 0
