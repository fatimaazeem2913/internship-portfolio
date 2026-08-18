"""
ingest_txt.py
-------------
Plain TXT ingestion.

TXT has neither pages (like PDF) nor style-based headings (like DOCX), but
this project's real TXT source document (api_rate_limiting_policy.txt)
does use a consistent convention: section headings are written in ALL CAPS
on their own line. This module detects that convention heuristically so
TXT chunks can still carry a meaningful "section heading" metadata field,
rather than being the one document type stuck with no structure at all.

This is a deliberately honest heuristic, not a guarantee -- it's documented
as such in docs/chunking_tradeoffs.md, since a plain TXT file with no
consistent heading convention would not benefit from this at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TxtBlock:
    source: str
    block_index: int
    text: str
    section_heading: str


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    # Heuristic: short-ish line, no terminal punctuation, and either fully
    # uppercase or title-cased with no lowercase sentence structure.
    if len(stripped) > 80:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    return stripped.isupper()


def extract_txt(path: str) -> list[TxtBlock]:
    source = Path(path).name
    raw_text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.split("\n")

    blocks: list[TxtBlock] = []
    current_heading = "Document Start"
    paragraph_buffer: list[str] = []
    idx = 0

    def flush_buffer():
        nonlocal idx
        if paragraph_buffer:
            text = " ".join(paragraph_buffer).strip()
            if text:
                blocks.append(TxtBlock(source, idx, text, current_heading))
                idx += 1
            paragraph_buffer.clear()

    for line in lines:
        if _looks_like_heading(line):
            flush_buffer()
            current_heading = line.strip()
            continue
        if line.strip() == "":
            flush_buffer()
        else:
            paragraph_buffer.append(line.strip())
    flush_buffer()

    return blocks


if __name__ == "__main__":
    blocks = extract_txt("data/txt/api_rate_limiting_policy.txt")
    print(f"Extracted {len(blocks)} blocks from TXT.\n")
    for b in blocks[:6]:
        print(f"(section: {b.section_heading}) {b.text[:90]}")
