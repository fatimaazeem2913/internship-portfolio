"""
ingest_pdf.py
-------------
PDF ingestion using two libraries, as required by the task spec:
  - pdfplumber (primary) -- generally better at layout-aware extraction
  - PyPDF2 (secondary/fallback + comparison) -- faster, simpler, weaker on
    complex layouts

Returns a list of PageRecord objects, one per page, each carrying the
metadata needed downstream (source filename, page number) so no chunk ever
loses its provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import PyPDF2


@dataclass
class PageRecord:
    source: str          # filename, e.g. "employee_handbook.pdf"
    page_number: int      # 1-indexed
    text: str
    extractor: str        # "pdfplumber" or "pypdf2"


def extract_with_pdfplumber(path: str) -> list[PageRecord]:
    source = Path(path).name
    records = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            records.append(PageRecord(source=source, page_number=i, text=text, extractor="pdfplumber"))
    return records


def extract_with_pypdf2(path: str) -> list[PageRecord]:
    source = Path(path).name
    records = []
    reader = PyPDF2.PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        records.append(PageRecord(source=source, page_number=i, text=text, extractor="pypdf2"))
    return records


def extract_pdf(path: str, prefer: str = "pdfplumber") -> list[PageRecord]:
    """Extract with the preferred library; fall back to the other if the
    preferred one yields suspiciously little text (e.g. <5 chars on a page
    that isn't actually blank), mirroring the real fallback pattern used
    in Day 15's ingestion.py."""
    primary = extract_with_pdfplumber(path) if prefer == "pdfplumber" else extract_with_pypdf2(path)
    fallback_fn = extract_with_pypdf2 if prefer == "pdfplumber" else extract_with_pdfplumber

    total_chars = sum(len(r.text.strip()) for r in primary)
    if total_chars < 20 * len(primary):  # heuristic: <20 chars/page average is suspicious
        fallback = fallback_fn(path)
        fallback_chars = sum(len(r.text.strip()) for r in fallback)
        if fallback_chars > total_chars:
            print(f"[ingest_pdf] WARNING: {prefer} extracted little text from {path}; "
                  f"using fallback extractor instead ({fallback_chars} vs {total_chars} chars).")
            return fallback
    return primary


if __name__ == "__main__":
    records = extract_pdf("data/pdfs/employee_handbook.pdf")
    print(f"Extracted {len(records)} pages via pdfplumber (primary).")
    print(f"Total characters: {sum(len(r.text) for r in records)}")
    print(f"\nPage 1 preview:\n{records[0].text[:300]}")

    # Compare both extractors head-to-head on the same file
    plumber = extract_with_pdfplumber("data/pdfs/employee_handbook.pdf")
    pypdf2_records = extract_with_pypdf2("data/pdfs/employee_handbook.pdf")
    print(f"\npdfplumber total chars: {sum(len(r.text) for r in plumber)}")
    print(f"PyPDF2 total chars:     {sum(len(r.text) for r in pypdf2_records)}")
