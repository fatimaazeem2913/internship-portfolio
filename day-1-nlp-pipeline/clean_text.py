"""
clean_text.py
-------------
Cleans raw scraped text: strips HTML tags, Markdown syntax, special characters, and numbers.
"""

import re


def strip_html(text: str) -> str:
    """Remove HTML tags like <p>, <b>, <h1> etc."""
    return re.sub(r"<[^>]+>", " ", text)


def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax: #, **bold**, *italic*, etc."""
    text = re.sub(r"#{1,6}\s*", "", text)          # headers ## Title
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)       # *italic*
    text = re.sub(r"_(.*?)_", r"\1", text)         # _italic_
    return text


def strip_numbers(text: str) -> str:
    """Remove standalone digits/numbers."""
    return re.sub(r"\d+", "", text)


def strip_special_chars(text: str) -> str:
    """Remove special characters, keep basic punctuation for sentence boundaries."""
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_pipeline(text: str) -> str:
    text = strip_html(text)
    text = strip_markdown(text)
    text = strip_numbers(text)
    text = strip_special_chars(text)
    text = normalize_whitespace(text)
    return text


if __name__ == "__main__":
    with open("data/input_corpus.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_pipeline(raw)

    with open("outputs/cleaned_corpus.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)

    print("--- BEFORE (first 300 chars) ---")
    print(raw[:300])
    print("\n--- AFTER (first 300 chars) ---")
    print(cleaned[:300])
    print("\nFull cleaned text saved to outputs/cleaned_corpus.txt")
