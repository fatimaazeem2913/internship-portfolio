"""
llm.py
------
Augmented Prompt -> LLM -> Grounded Response.

Follows the same USE_MOCK_LLM pattern established from Day 11 onward:
  USE_MOCK_LLM=true  -> deterministic offline mock (no network/API key
                         needed, used for free/fast testing and CI).
  USE_MOCK_LLM unset/false -> real call to Gemini via the google-genai SDK
                         (the project's primary LLM since Day 8's OpenAI
                         billing wall).

Real bug this module deliberately guards against (documented from Day 9,
resurfaced Day 14, referenced again in original Day 15): Gemini 3.x models
silently ignore the `temperature` parameter -- it does not error, it just
has no effect. Setting temperature=0.0 here for grounded RAG answers (which
should be deterministic and non-creative) is honest about NOT being a real
fix for that bug, since it happens to want temperature=0 anyway and would
look "fixed" whether or not the underlying bug is really gone. Anywhere
this project genuinely needs controlled randomness (which grounded RAG
answers do not), the correct fix remains injecting variation into the
prompt text itself, per the Day 9 finding -- not the temperature parameter.
"""

from __future__ import annotations

import os

GEMINI_MODEL = "gemini-3.5-flash-lite"


def _use_mock() -> bool:
    # Read fresh on every call rather than caching at import time -- caching
    # at import broke this exact module the first time it was tested here:
    # setting os.environ inside `if __name__ == "__main__"` ran AFTER the
    # module-level assignment had already executed, so the mock flag never
    # took effect. Reading live avoids that class of bug entirely.
    return os.environ.get("USE_MOCK_LLM", "false").lower() == "true"

RAG_SYSTEM_PROMPT = """You are a support assistant that answers questions ONLY using the provided context.

Rules:
- Answer using ONLY information present in the context below.
- If the context does not contain enough information to answer, say so explicitly -- do not guess or use outside knowledge.
- Cite which source(s) you used by filename.
- Keep answers concise and direct.
"""


def build_augmented_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for i, hit in enumerate(retrieved_chunks, start=1):
        source = os.path.basename(hit["metadata"].get("source", "unknown"))
        context_blocks.append(f"[Context {i} — source: {source}]\n{hit['text']}")

    context_text = "\n\n".join(context_blocks)
    return f"{RAG_SYSTEM_PROMPT}\n\nCONTEXT:\n{context_text}\n\nQUESTION: {query}\n\nANSWER:"


def _mock_generate(prompt: str, query: str, retrieved_chunks: list[dict]) -> str:
    """Deterministic offline stand-in: extracts the single retrieved chunk
    with the strongest match signal and returns it as a templated answer,
    rather than faking free-form generation. This keeps mock mode honest
    about being a mock -- no pretending an LLM 'reasoned' about anything."""
    if not retrieved_chunks:
        return "I don't have enough information in the provided context to answer that."

    top = retrieved_chunks[0]
    source = os.path.basename(top["metadata"].get("source", "unknown"))
    return (
        f"[MOCK RESPONSE] Based on the most relevant retrieved context "
        f"(source: {source}): {top['text'].strip()}"
    )


def _real_generate(prompt: str) -> str:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a free key at aistudio.google.com/apikey "
            "and export it, or set USE_MOCK_LLM=true to run without one."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        # temperature=0.0 here is a design choice for grounded RAG answers,
        # NOT a workaround for the Day 9 temperature-ignored bug -- see
        # module docstring.
        config={"temperature": 0.0},
    )
    return response.text


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    prompt = build_augmented_prompt(query, retrieved_chunks)

    if _use_mock():
        answer = _mock_generate(prompt, query, retrieved_chunks)
        backend = "mock"
    else:
        answer = _real_generate(prompt)
        backend = GEMINI_MODEL

    sources = sorted({os.path.basename(h["metadata"].get("source", "unknown")) for h in retrieved_chunks})
    return {"answer": answer, "sources": sources, "backend": backend, "prompt": prompt}


if __name__ == "__main__":
    # Runs in mock mode by default so this file is testable without any API
    # key or network access, matching the established free/offline-first
    # testing pattern from Day 11 onward.
    os.environ.setdefault("USE_MOCK_LLM", "true")

    from ingestion import load_corpus
    from chunking import chunk_documents
    from vector_store import VectorStore
    from retrieval import Retriever

    docs = load_corpus("data/corpus")
    chunks = chunk_documents(docs)
    store = VectorStore(persist_dir="data/vector_store_test")
    store.reset()
    store.add_chunks(chunks)
    retriever = Retriever(chunks, store)

    query = "How long do I have to report a damaged item?"
    hits = retriever.hybrid_retrieve(query, top_k=3)
    result = generate_answer(query, hits)

    print(f"Query: {query}")
    print(f"Backend: {result['backend']}")
    print(f"Sources: {result['sources']}")
    print(f"\nAnswer:\n{result['answer']}")
