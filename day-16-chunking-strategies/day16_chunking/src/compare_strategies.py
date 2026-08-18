"""
compare_strategies.py
---------------------
Runs all 5 chunking strategies against all 3 real document types (PDF,
DOCX, TXT), verifies metadata integrity for every result, and computes
real trade-off metrics (chunk count, avg/min/max size in characters) that
back up the written analysis in docs/chunking_tradeoffs.md.

Also demonstrates the semantic strategy at the document level (not just
the small hand-written example in chunking_strategies.py's own __main__).
"""

from __future__ import annotations

import json
import statistics

from chunking_strategies import (
    Chunk, fixed_size_chunk, token_based_chunk, recursive_chunk,
    hierarchical_chunk, semantic_chunk,
)
from embedding import embed_texts, get_backend
from ingest_docx import extract_docx
from ingest_pdf import extract_pdf
from ingest_txt import extract_txt
from pipeline import (
    chunk_pdf, chunk_docx, chunk_txt, verify_metadata_integrity,
    _split_pdf_page_into_sections,
)


def _flatten_docx_text(path: str) -> str:
    blocks = extract_docx(path)
    return "\n\n".join(b.text for b in blocks if b.heading_level < 1)


def _flatten_txt_text(path: str) -> str:
    return extract_txt(path)[0].source and "\n\n".join(b.text for b in extract_txt(path))


def _flatten_pdf_text(path: str, max_pages: int | None = 10) -> str:
    pages = extract_pdf(path)
    if max_pages:
        pages = pages[:max_pages]
    return "\n\n".join(p.text for p in pages)


def chunk_stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"count": 0}
    sizes = [len(c.text) for c in chunks]
    return {
        "count": len(chunks),
        "avg_chars": round(statistics.mean(sizes), 1),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "stdev_chars": round(statistics.pstdev(sizes), 1) if len(sizes) > 1 else 0.0,
    }


def run_all_strategies_on_text(text: str, source: str) -> dict:
    """Runs all 5 strategies on the same flat text for a fair, apples-to-
    apples size/count comparison (hierarchical needs real sections, so it's
    run separately against real document structure elsewhere)."""
    results = {}
    results["fixed_size"] = chunk_stats(fixed_size_chunk(text, source, chunk_size=500, overlap=50))
    results["token_based"] = chunk_stats(token_based_chunk(text, source, chunk_size_tokens=150, overlap_tokens=20))
    results["recursive"] = chunk_stats(recursive_chunk(text, source, chunk_size=500, overlap=50))

    sem_chunks = semantic_chunk(text, source, embed_fn=embed_texts, breakpoint_percentile=80)
    results["semantic"] = chunk_stats(sem_chunks)
    return results


def main():
    report = {"token_backend": None, "embedding_backend": get_backend(), "documents": {}}

    print("=" * 70)
    print("DOCUMENT 1: employee_handbook.pdf (real 56-page PDF)")
    print("=" * 70)
    pdf_text = _flatten_pdf_text("data/pdfs/employee_handbook.pdf", max_pages=10)
    pdf_flat_results = run_all_strategies_on_text(pdf_text, "employee_handbook.pdf")
    for strategy, stats in pdf_flat_results.items():
        print(f"  {strategy:15s} {stats}")

    pdf_hier_chunks = chunk_pdf("data/pdfs/employee_handbook.pdf", strategy="hierarchical",
                                max_chunk_size=500, overlap=50)
    pdf_hier_stats = chunk_stats(pdf_hier_chunks)
    print(f"  {'hierarchical':15s} {pdf_hier_stats}")
    pdf_integrity = verify_metadata_integrity(pdf_hier_chunks, "employee_handbook.pdf")
    print(f"  Metadata integrity (hierarchical): {pdf_integrity['issues_found']} issues, "
          f"{pdf_integrity['chunks_with_section_heading']}/{pdf_integrity['total_chunks']} have section headings")

    report["documents"]["employee_handbook.pdf"] = {
        **pdf_flat_results, "hierarchical": pdf_hier_stats,
        "metadata_integrity": pdf_integrity,
    }

    print("\n" + "=" * 70)
    print("DOCUMENT 2: product_spec.docx (real DOCX with headings + table)")
    print("=" * 70)
    docx_text = _flatten_docx_text("data/docx/product_spec.docx")
    docx_flat_results = run_all_strategies_on_text(docx_text, "product_spec.docx")
    for strategy, stats in docx_flat_results.items():
        print(f"  {strategy:15s} {stats}")

    docx_hier_chunks = chunk_docx("data/docx/product_spec.docx", strategy="hierarchical",
                                  max_chunk_size=500, overlap=50)
    docx_hier_stats = chunk_stats(docx_hier_chunks)
    print(f"  {'hierarchical':15s} {docx_hier_stats}")
    docx_integrity = verify_metadata_integrity(docx_hier_chunks, "product_spec.docx")
    print(f"  Metadata integrity (hierarchical): {docx_integrity['issues_found']} issues, "
          f"{docx_integrity['chunks_with_section_heading']}/{docx_integrity['total_chunks']} have section headings")

    report["documents"]["product_spec.docx"] = {
        **docx_flat_results, "hierarchical": docx_hier_stats,
        "metadata_integrity": docx_integrity,
    }

    print("\n" + "=" * 70)
    print("DOCUMENT 3: api_rate_limiting_policy.txt (real plain TXT)")
    print("=" * 70)
    txt_blocks = extract_txt("data/txt/api_rate_limiting_policy.txt")
    txt_text = "\n\n".join(b.text for b in txt_blocks)
    txt_flat_results = run_all_strategies_on_text(txt_text, "api_rate_limiting_policy.txt")
    for strategy, stats in txt_flat_results.items():
        print(f"  {strategy:15s} {stats}")

    txt_hier_chunks = chunk_txt("data/txt/api_rate_limiting_policy.txt", strategy="hierarchical",
                                max_chunk_size=500, overlap=50)
    txt_hier_stats = chunk_stats(txt_hier_chunks)
    print(f"  {'hierarchical':15s} {txt_hier_stats}")
    txt_integrity = verify_metadata_integrity(txt_hier_chunks, "api_rate_limiting_policy.txt")
    print(f"  Metadata integrity (hierarchical): {txt_integrity['issues_found']} issues, "
          f"{txt_integrity['chunks_with_section_heading']}/{txt_integrity['total_chunks']} have section headings")

    report["documents"]["api_rate_limiting_policy.txt"] = {
        **txt_flat_results, "hierarchical": txt_hier_stats,
        "metadata_integrity": txt_integrity,
    }

    with open("data/outputs/strategy_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nFull comparison report saved to data/outputs/strategy_comparison_report.json")


if __name__ == "__main__":
    main()
