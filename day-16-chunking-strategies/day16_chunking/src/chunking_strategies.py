"""
chunking_strategies.py
----------------------
Implements and lets you directly compare all 5 chunking strategies required
by the task:

  1. Fixed-size       -- naive character-count windows, fixed overlap
  2. Token-based       -- windows sized by actual LLM token count (tiktoken),
                          not character/word count, since token count is
                          what actually determines whether a chunk fits in
                          a model's context window
  3. Recursive         -- LangChain's RecursiveCharacterTextSplitter, which
                          tries a hierarchy of separators (paragraph -> line
                          -> sentence -> word) before falling back to a hard
                          character cut
  4. Semantic          -- splits at points where adjacent sentences' meaning
                          diverges most, using real sentence-transformers
                          embeddings and cosine-distance breakpoints
  5. Hierarchical      -- preserves document structure explicitly: splits
                          along heading boundaries first (chapter/section),
                          THEN applies recursive splitting only within a
                          section if that section is still too long

Every chunk produced by every strategy carries a Chunk object with full
metadata (source, page_number, chunk_index, section_heading) -- see
metadata.py for the shared schema all 5 strategies write into.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    strategy: str
    source: str
    chunk_index: int
    page_number: int | None = None
    section_heading: str | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Fixed-size chunking
# ---------------------------------------------------------------------------

def fixed_size_chunk(text: str, source: str, chunk_size: int = 500, overlap: int = 50,
                      page_number: int | None = None, section_heading: str | None = None) -> list[Chunk]:
    """Naive character-count windows. Simple, fast, and completely blind to
    sentence or paragraph boundaries -- a chunk can start or end mid-word."""
    chunks = []
    start = 0
    idx = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        window = text[start:start + chunk_size]
        if window.strip():
            chunks.append(Chunk(window, "fixed_size", source, idx, page_number, section_heading))
            idx += 1
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# 2. Token-based chunking
# ---------------------------------------------------------------------------

_ENCODER = None
_TOKEN_BACKEND = None  # "tiktoken" or "word-approx-fallback"


def _get_encoder():
    """Lazy singleton, mirroring Day 15's embedding.py fallback pattern.

    Real, honestly-documented finding: tiktoken's cl100k_base encoding file
    is downloaded on first use from openaipublic.blob.core.windows.net,
    which is NOT in this sandbox's network allowlist -- the download fails
    with a 403 from the egress proxy. Same category of issue as Day 15's
    huggingface.co / sentence-transformers finding: a component silently
    depends on runtime network access to a file that isn't bundled with the
    package itself.

    Fix, mirroring how that earlier bug was handled: the real tiktoken path
    is kept as the correct, primary implementation (this is what runs on a
    normal machine with real internet access -- e.g. the user's own local
    machine). For sandbox testing, a word-count-based approximation is used
    instead, clearly logged as a fallback, never silently passed off as
    real token counts.
    """
    global _ENCODER, _TOKEN_BACKEND
    if _ENCODER is not None or _TOKEN_BACKEND is not None:
        return _ENCODER

    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
        _TOKEN_BACKEND = "tiktoken"
        print("[chunking_strategies] Loaded real tiktoken cl100k_base encoding.")
    except Exception as exc:
        _ENCODER = None
        _TOKEN_BACKEND = "word-approx-fallback"
        print(
            f"[chunking_strategies] WARNING: could not load tiktoken cl100k_base "
            f"({type(exc).__name__}: openaipublic.blob.core.windows.net unreachable "
            f"in this sandbox). Falling back to a word-count-based token "
            f"approximation (~0.75 tokens/word) for this run -- NOT real token "
            f"counts. Run this on a machine with real internet access for the "
            f"genuine tiktoken-based result."
        )
    return _ENCODER


def _encode(text: str) -> list:
    """Returns a list whose LENGTH is meaningful (token/pseudo-token count)
    even in fallback mode; the list's contents are only ever decoded back
    via _decode() below, never inspected directly, so the fallback's
    "tokens" don't need to correspond to anything except a stable,
    invertible word-based split."""
    encoder = _get_encoder()
    if _TOKEN_BACKEND == "tiktoken":
        return encoder.encode(text)
    return text.split(" ")  # fallback: "tokens" are just whitespace-split words


def _decode(tokens: list) -> str:
    if _TOKEN_BACKEND == "tiktoken":
        return _get_encoder().decode(tokens)
    return " ".join(tokens)


def get_token_backend() -> str:
    _get_encoder()
    return _TOKEN_BACKEND


def token_based_chunk(text: str, source: str, chunk_size_tokens: int = 150, overlap_tokens: int = 20,
                       page_number: int | None = None, section_heading: str | None = None) -> list[Chunk]:
    """Windows sized by real token count, not characters. This matters
    because a model's context window is a token budget, not a character
    budget -- a 500-character chunk of dense technical text and a
    500-character chunk of simple prose can differ by 30%+ in token count."""
    tokens = _encode(text)
    chunks = []
    start = 0
    idx = 0
    step = max(chunk_size_tokens - overlap_tokens, 1)
    while start < len(tokens):
        window_tokens = tokens[start:start + chunk_size_tokens]
        window_text = _decode(window_tokens)
        if window_text.strip():
            chunks.append(Chunk(window_text, "token_based", source, idx, page_number, section_heading,
                                 metadata={"token_count": len(window_tokens), "token_backend": get_token_backend()}))
            idx += 1
        if start + chunk_size_tokens >= len(tokens):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# 3. Recursive chunking (LangChain)
# ---------------------------------------------------------------------------

def recursive_chunk(text: str, source: str, chunk_size: int = 500, overlap: int = 50,
                     page_number: int | None = None, section_heading: str | None = None) -> list[Chunk]:
    """LangChain's RecursiveCharacterTextSplitter: tries splitting on
    paragraph breaks first, then line breaks, then sentences, then words,
    only falling back to a hard character cut if nothing else fits --
    meaning most chunks end up respecting natural language boundaries
    without any manual heading-detection logic."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    return [
        Chunk(piece, "recursive", source, i, page_number, section_heading)
        for i, piece in enumerate(pieces) if piece.strip()
    ]


# ---------------------------------------------------------------------------
# 4. Semantic chunking
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    # Simple, dependency-free sentence splitter: good enough for this
    # project's real corpus (formal policy/spec prose), not a full NLP
    # sentence boundary detector.
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def semantic_chunk(text: str, source: str, embed_fn, breakpoint_percentile: float = 85.0,
                    page_number: int | None = None, section_heading: str | None = None) -> list[Chunk]:
    """Splits at the sentence-to-sentence transitions with the LARGEST
    semantic distance -- i.e. wherever the topic actually shifts the most,
    rather than at any fixed size. Requires an embedding function (real
    sentence-transformers or the honest fallback, matching Day 15's
    embedding.py pattern) to be passed in.

    Algorithm:
      1. Split into sentences.
      2. Embed every sentence.
      3. Compute cosine distance between each consecutive sentence pair.
      4. Treat distances above the given percentile as "topic breaks" and
         cut the chunk there.
    """
    import numpy as np

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [Chunk(text, "semantic", source, 0, page_number, section_heading)] if text.strip() else []

    vectors = np.array(embed_fn(sentences))
    distances = []
    for i in range(len(vectors) - 1):
        a, b = vectors[i], vectors[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        distances.append(1 - cos_sim)  # cosine distance

    if not distances:
        return [Chunk(text, "semantic", source, 0, page_number, section_heading)]

    threshold = np.percentile(distances, breakpoint_percentile)
    breakpoints = {i for i, d in enumerate(distances) if d >= threshold}

    chunks = []
    current: list[str] = []
    idx = 0
    for i, sentence in enumerate(sentences):
        current.append(sentence)
        if i in breakpoints:
            chunk_text = " ".join(current)
            chunks.append(Chunk(chunk_text, "semantic", source, idx, page_number, section_heading,
                                 metadata={"num_sentences": len(current)}))
            idx += 1
            current = []
    if current:
        chunk_text = " ".join(current)
        chunks.append(Chunk(chunk_text, "semantic", source, idx, page_number, section_heading,
                             metadata={"num_sentences": len(current)}))
    return chunks


# ---------------------------------------------------------------------------
# 5. Hierarchical chunking
# ---------------------------------------------------------------------------

def hierarchical_chunk(sections: list[tuple[str, str]], source: str, max_chunk_size: int = 500,
                        overlap: int = 50) -> list[Chunk]:
    """Takes a list of (section_heading, section_text) pairs -- i.e. the
    document's OWN real structure, already parsed out by the ingestion
    layer (heading levels for DOCX, ALL-CAPS headings for TXT, chapter
    titles for PDF) -- and splits WITHIN each section only if that
    section's text still exceeds max_chunk_size. A short section becomes
    exactly one chunk; a long section gets recursively split, but the
    section boundary itself is never crossed.

    This is the one strategy that requires structure-aware ingestion
    upstream -- it can't be applied to raw undifferentiated text the way
    the other four can."""
    chunks = []
    idx = 0
    for heading, text in sections:
        if len(text) <= max_chunk_size:
            chunks.append(Chunk(text, "hierarchical", source, idx, section_heading=heading,
                                 metadata={"section_split": False}))
            idx += 1
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""],
            )
            pieces = splitter.split_text(text)
            for piece in pieces:
                if piece.strip():
                    chunks.append(Chunk(piece, "hierarchical", source, idx, section_heading=heading,
                                         metadata={"section_split": True}))
                    idx += 1
    return chunks


if __name__ == "__main__":
    sample_text = (
        "Free-tier API keys are limited to 60 requests per minute and 10,000 requests per day, "
        "calculated on a rolling window basis rather than a fixed calendar window. Professional-tier "
        "API keys are limited to 600 requests per minute and 200,000 requests per day. "
        "Enterprise-tier customers negotiate custom limits directly with their account manager.\n\n"
        "Every API response includes three headers describing the caller's current rate limit status: "
        "X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset. Clients should treat these "
        "headers as authoritative and adjust request pacing accordingly."
    )

    print("=== Fixed-size (200 chars, 30 overlap) ===")
    for c in fixed_size_chunk(sample_text, "sample.txt", chunk_size=200, overlap=30):
        print(f"  [{c.chunk_index}] ({len(c.text)} chars) {c.text[:60]!r}")

    print("\n=== Token-based (40 tokens, 5 overlap) ===")
    for c in token_based_chunk(sample_text, "sample.txt", chunk_size_tokens=40, overlap_tokens=5):
        print(f"  [{c.chunk_index}] ({c.metadata['token_count']} tokens) {c.text[:60]!r}")

    print("\n=== Recursive (200 chars, 30 overlap) ===")
    for c in recursive_chunk(sample_text, "sample.txt", chunk_size=200, overlap=30):
        print(f"  [{c.chunk_index}] ({len(c.text)} chars) {c.text[:60]!r}")

    print("\n=== Hierarchical (2 sections) ===")
    sections = [
        ("STANDARD RATE LIMITS", sample_text.split("\n\n")[0]),
        ("RATE LIMIT HEADERS", sample_text.split("\n\n")[1]),
    ]
    for c in hierarchical_chunk(sections, "sample.txt", max_chunk_size=200, overlap=30):
        print(f"  [{c.chunk_index}] (section: {c.section_heading}) ({len(c.text)} chars) {c.text[:60]!r}")
