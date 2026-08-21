import re
from typing import List
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .models import DocumentElement, Chunk

def chunk_fixed_size(elements: List[DocumentElement], chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    chunks = []
    idx = 0
    for el in elements:
        text = el.content
        start = 0
        while start < len(text):
            chunk_str = text[start:start + chunk_size]
            meta = el.metadata.copy()
            meta.update({"chunk_index": idx, "strategy": "fixed_size"})
            chunks.append(Chunk(chunk_id=f"{meta['source']}_FX_{idx}", content=chunk_str, metadata=meta))
            idx += 1
            start += (chunk_size - overlap)
            if start >= len(text) or chunk_size <= overlap:
                break
    return chunks

def chunk_token_based(elements: List[DocumentElement], max_tokens: int = 150, overlap_tokens: int = 25) -> List[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    idx = 0
    for el in elements:
        tokens = enc.encode(el.content)
        start = 0
        while start < len(tokens):
            chunk_tokens = tokens[start:start + max_tokens]
            chunk_str = enc.decode(chunk_tokens)
            meta = el.metadata.copy()
            meta.update({"chunk_index": idx, "strategy": "token_based", "token_count": len(chunk_tokens)})
            chunks.append(Chunk(chunk_id=f"{meta['source']}_TK_{idx}", content=chunk_str, metadata=meta))
            idx += 1
            start += (max_tokens - overlap_tokens)
            if start >= len(tokens) or max_tokens <= overlap_tokens:
                break
    return chunks

def chunk_recursive(elements: List[DocumentElement], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", r"(?<=\.) ", " ", ""]
    )
    # ... rest of the function remains the same
    chunks = []
    idx = 0
    for el in elements:
        splits = splitter.split_text(el.content)
        for split in splits:
            meta = el.metadata.copy()
            meta.update({"chunk_index": idx, "strategy": "recursive"})
            chunks.append(Chunk(chunk_id=f"{meta['source']}_RC_{idx}", content=split, metadata=meta))
            idx += 1
    return chunks

def chunk_semantic(elements: List[DocumentElement], max_sentences: int = 3) -> List[Chunk]:
    chunks = []
    idx = 0
    for el in elements:
        sentences = re.split(r'(?<=[.?!])\s+', el.content)
        current = []
        for sent in sentences:
            if sent.strip():
                current.append(sent.strip())
            if len(current) >= max_sentences:
                meta = el.metadata.copy()
                meta.update({"chunk_index": idx, "strategy": "semantic"})
                chunks.append(Chunk(chunk_id=f"{meta['source']}_SM_{idx}", content=" ".join(current), metadata=meta))
                idx += 1
                current = []
        if current:
            meta = el.metadata.copy()
            meta.update({"chunk_index": idx, "strategy": "semantic"})
            chunks.append(Chunk(chunk_id=f"{meta['source']}_SM_{idx}", content=" ".join(current), metadata=meta))
            idx += 1
    return chunks

def chunk_hierarchical(elements: List[DocumentElement], parent_size: int = 800, child_size: int = 200) -> List[Chunk]:
    chunks = []
    idx = 0
    p_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=0)
    c_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=25)
    for el in elements:
        parent_docs = p_splitter.split_text(el.content)
        for p_idx, p_text in enumerate(parent_docs):
            p_id = f"{el.metadata['source']}_P{p_idx}"
            child_docs = c_splitter.split_text(p_text)
            for c_text in child_docs:
                meta = el.metadata.copy()
                meta.update({
                    "chunk_index": idx,
                    "strategy": "hierarchical",
                    "parent_id": p_id
                })
                chunks.append(Chunk(
                    chunk_id=f"{meta['source']}_HC_{idx}",
                    content=c_text,
                    metadata=meta,
                    parent_id=p_id
                ))
                idx += 1
    return chunks
