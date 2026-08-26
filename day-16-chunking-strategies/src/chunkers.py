import re
from typing import List, Dict, Any

class BaseChunker:
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

class FixedWindowChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            segment = text[start:end]
            if segment.strip():
                chunks.append({
                    "chunk_id": f"fixed_{metadata.get("page_number", 1)}_{idx:03d}",
                    "content": segment.strip(),
                    "metadata": {**metadata, "chunk_type": "fixed", "start_char": start, "end_char": end}
                })
                idx += 1
            start += (self.chunk_size - self.overlap)
        return chunks

class SentenceChunker(BaseChunker):
    def __init__(self, sentences_per_chunk: int = 3, sentence_overlap: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        sentences = re.split(r"(?<=[.?!])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        idx = 0
        step = max(1, self.sentences_per_chunk - self.sentence_overlap)
        for i in range(0, len(sentences), step):
            batch = sentences[i:i + self.sentences_per_chunk]
            if batch:
                content = " ".join(batch)
                chunks.append({
                    "chunk_id": f"sentence_{metadata.get("page_number", 1)}_{idx:03d}",
                    "content": content,
                    "metadata": {**metadata, "chunk_type": "sentence", "sentence_count": len(batch)}
                })
                idx += 1
        return chunks

class RecursiveSemanticChunker(BaseChunker):
    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 40):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        paragraphs = re.split(r"\n{2,}", text)
        chunks = []
        idx = 0
        curr_text = ""
        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            if len(curr_text.split()) + len(p_clean.split()) <= self.max_tokens:
                curr_text = f"{curr_text}\n\n{p_clean}".strip()
            else:
                if curr_text:
                    chunks.append({
                        "chunk_id": f"semantic_{metadata.get("page_number", 1)}_{idx:03d}",
                        "content": curr_text,
                        "metadata": {**metadata, "chunk_type": "semantic"}
                    })
                    idx += 1
                curr_text = p_clean
        if curr_text:
            chunks.append({
                "chunk_id": f"semantic_{metadata.get("page_number", 1)}_{idx:03d}",
                "content": curr_text,
                "metadata": {**metadata, "chunk_type": "semantic"}
            })
        return chunks

class HierarchicalParentChildChunker:
    def __init__(self, parent_size: int = 1000, child_size: int = 250, child_overlap: int = 30):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        all_chunks = []
        start = 0
        parent_idx = 0
        while start < len(text):
            p_end = start + self.parent_size
            parent_text = text[start:p_end].strip()
            if not parent_text:
                break
            
            p_id = f"parent_{metadata.get("page_number", 1)}_{parent_idx:03d}"
            all_chunks.append({
                "chunk_id": p_id,
                "content": parent_text,
                "parent_id": None,
                "metadata": {**metadata, "chunk_type": "parent"}
            })

            # Create Child Chunks linked to parent
            c_start = 0
            child_idx = 0
            while c_start < len(parent_text):
                c_end = c_start + self.child_size
                child_segment = parent_text[c_start:c_end].strip()
                if child_segment:
                    all_chunks.append({
                        "chunk_id": f"child_{p_id}_{child_idx:02d}",
                        "content": child_segment,
                        "parent_id": p_id,
                        "metadata": {**metadata, "chunk_type": "child", "parent_id": p_id}
                    })
                    child_idx += 1
                c_start += (self.child_size - self.child_overlap)

            parent_idx += 1
            start += self.parent_size
        return all_chunks
