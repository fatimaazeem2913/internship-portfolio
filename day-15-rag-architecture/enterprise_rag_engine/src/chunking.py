"""
chunking.py
-----------
Ingestion -> Chunking.

Splits Document objects into overlapping, section-aware chunks.

Strategy: recursive/hierarchical splitting.
  1. Try to split on double-newlines (paragraph boundaries) first, since
     these policy documents are section-structured and splitting mid-
     paragraph tends to separate a rule from its own qualifying clause
     (this is exactly the "poor chunking" RAG failure mode documented in
     rag_failure_modes.md -- splitting "Refunds are processed within 5-7
     business days" from the very next sentence about payment method
     fallback would make retrieval return an incomplete rule).
  2. If a paragraph is still longer than chunk_size, fall back to a
     sliding window over words with overlap, so no chunk exceeds the
     target size while still preserving some cross-chunk context.

Each chunk keeps a reference back to its source document and its index,
which the pipeline later uses for citation ("Source: refund_policy.txt").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ingestion import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_paragraph_by_words(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    words = paragraph.split()
    if len(words) <= chunk_size:
        return [paragraph]

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)  # guard against overlap >= chunk_size
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_document(doc: Document, chunk_size: int = 120, overlap: int = 20) -> list[Chunk]:
    """
    chunk_size / overlap are measured in words, not characters or tokens.
    Word-count is an approximation of token count (roughly 0.75 tokens per
    word for English), which is precise enough for chunking purposes --
    exact tokenization is the embedding model's job, not the chunker's.
    """
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    if not paragraphs:
        # No blank-line structure found (common in PDF-extracted text where
        # paragraph breaks collapse into single newlines) -- fall back to
        # treating the whole document as one block for the word-window split.
        paragraphs = [doc.text.strip()]

    raw_chunks: list[str] = []
    for para in paragraphs:
        raw_chunks.extend(_split_paragraph_by_words(para, chunk_size, overlap))

    chunks = [
        Chunk(
            chunk_id=f"{doc.doc_id}_chunk{i}",
            doc_id=doc.doc_id,
            source=doc.source,
            text=text,
            metadata={**doc.metadata, "chunk_index": i},
        )
        for i, text in enumerate(raw_chunks)
    ]
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int = 120, overlap: int = 20) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks


if __name__ == "__main__":
    from ingestion import load_corpus

    docs = load_corpus("data/corpus")
    chunks = chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks[:3]:
        print(f"[{c.chunk_id}] ({len(c.text.split())} words)")
        print(c.text[:150].replace("\n", " ") + "...\n")
