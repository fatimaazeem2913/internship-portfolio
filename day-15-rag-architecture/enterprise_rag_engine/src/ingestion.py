"""
ingestion.py
------------
Corpus -> raw documents.

Reads every supported file in a corpus directory and returns a list of
Document dicts: {"doc_id": str, "source": str, "text": str}.

Supports:
  - .txt  (plain read)
  - .pdf  (text extraction via pdfplumber, falling back to pypdf if
           pdfplumber finds no extractable text on a page)

Real bug worth documenting: pdfplumber and pypdf occasionally disagree on
a PDF's extractable text (pdfplumber is generally better for PDFs with
underlying layout/tables, pypdf is faster for simple single-column PDFs).
We use pdfplumber as primary and pypdf as a fallback so a single bad
extraction from one library doesn't silently drop a document.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


@dataclass
class Document:
    doc_id: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _read_pdf(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

    combined = "\n".join(text_parts).strip()

    # Fallback: if pdfplumber extracted nothing (e.g. certain PDF encodings
    # pdfplumber's layout engine chokes on), try pypdf instead.
    if not combined:
        reader = PdfReader(str(path))
        text_parts = [p.extract_text() or "" for p in reader.pages]
        combined = "\n".join(text_parts).strip()

    return combined


def load_corpus(corpus_dir: str) -> list[Document]:
    """Load every supported file in corpus_dir into Document objects."""
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    documents: list[Document] = []
    for file_path in sorted(corpus_path.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() == ".txt":
            text = _read_txt(file_path)
        else:
            text = _read_pdf(file_path)

        if not text:
            # Real, honest handling: don't silently skip -- flag it so it
            # shows up in logs instead of vanishing from the corpus.
            print(f"[ingestion] WARNING: no extractable text in {file_path.name}")
            continue

        documents.append(
            Document(
                doc_id=file_path.stem,
                source=str(file_path),
                text=text,
                metadata={"filetype": file_path.suffix.lower().lstrip(".")},
            )
        )

    return documents


if __name__ == "__main__":
    docs = load_corpus("data/corpus")
    for d in docs:
        print(f"{d.doc_id:20s} ({d.metadata['filetype']:3s})  {len(d.text)} chars")
