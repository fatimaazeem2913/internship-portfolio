"""
extract_chunk_clean.py
--------------------------
Stage 1 of the Day 7 mini-project: extract text from 3 PDFs (different
domains), chunk it, label each chunk by source document, then clean the
resulting corpus with stopword removal and lemmatization.

Uses PyMuPDF (imported as `fitz`) for extraction, as specified in the task.
"""

import re
import json
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

SOURCE_PDFS = {
    "research_paper": "pdfs/research_paper.pdf",
    "news_article": "pdfs/news_article.pdf",
    "technical_manual": "pdfs/technical_manual.pdf",
}


def extract_text_pymupdf(pdf_path):
    """Extract raw text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text, max_sentences_per_chunk=3):
    """
    Chunk text into groups of a few sentences each. Sentence-based chunking
    (rather than fixed character windows) keeps each chunk semantically
    coherent, which matters for retrieval quality later.
    """
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences_per_chunk):
        chunk = " ".join(sentences[i:i + max_sentences_per_chunk]).strip()
        if len(chunk) > 20:  # skip degenerate tiny fragments
            chunks.append(chunk)
    return chunks


def clean_text(text):
    """
    Clean a chunk of text: lowercase, remove non-alphabetic characters,
    tokenize, remove stopwords, and lemmatize each remaining token.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    lemmatized = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(lemmatized)


def build_corpus():
    """
    Extracts, chunks, labels, and cleans all 3 source PDFs into a single
    labeled corpus structure.
    """
    corpus = []
    chunk_id = 0

    for source_name, pdf_path in SOURCE_PDFS.items():
        raw_text = extract_text_pymupdf(pdf_path)
        chunks = chunk_text(raw_text)

        for chunk in chunks:
            corpus.append({
                "chunk_id": chunk_id,
                "source": source_name,
                "raw_text": chunk,
                "cleaned_text": clean_text(chunk),
            })
            chunk_id += 1

    return corpus


if __name__ == "__main__":
    print("=" * 90)
    print("STAGE 1: PDF EXTRACTION, CHUNKING, LABELING, AND CLEANING")
    print("=" * 90)

    corpus = build_corpus()

    print(f"\nTotal chunks extracted across all 3 sources: {len(corpus)}")
    by_source = {}
    for c in corpus:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    for source, count in by_source.items():
        print(f"  {source}: {count} chunks")

    print("\n--- Sample chunks (first from each source) ---\n")
    seen_sources = set()
    for c in corpus:
        if c["source"] not in seen_sources:
            seen_sources.add(c["source"])
            print(f"[chunk_id={c['chunk_id']}] source={c['source']}")
            print(f"  RAW:     {c['raw_text'][:150]}...")
            print(f"  CLEANED: {c['cleaned_text'][:150]}...")
            print()

    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    print(f"Full labeled, cleaned corpus saved to data/corpus.json ({len(corpus)} chunks)")
