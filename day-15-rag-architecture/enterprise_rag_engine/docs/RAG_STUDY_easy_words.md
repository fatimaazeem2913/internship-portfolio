# Day 15 — RAG, Explained in Easy Words

## What problem are we solving?

An LLM (like Gemini) only knows what it learned during training. Ask it
about your company's refund policy and it'll either say "I don't know" or,
worse, confidently make something up. RAG fixes this by giving the model
your real documents to read *right before* it answers, instead of relying
only on what's stuck in its memory.

Think of it like an open-book exam. Without RAG, the model is answering
from memory alone. With RAG, it gets to flip to the right page of the
textbook first.

## The pipeline, in plain words

1. **Corpus** — your pile of real documents (policy files, PDFs, whatever).
2. **Ingestion** — actually reading those files into plain text the
   computer can work with.
3. **Chunking** — cutting long documents into smaller pieces, like cutting
   a long book into paragraphs, so we don't hand the model the entire book
   every time — just the relevant page.
4. **Embedding** — turning each chunk of text into a list of numbers that
   captures its *meaning*. Similar meanings end up as similar numbers.
5. **Vector Store** — a filing cabinet that stores all those number-lists
   so we can search through them fast.
6. **Retrieval** — when someone asks a question, turn the question into
   numbers too, then find the chunks whose numbers are most similar.
7. **Augmented Prompt** — glue the question and the found chunks together
   into one message.
8. **LLM** — the model reads that message and writes an answer.
9. **Grounded Response** — the final answer, plus which document it came
   from — so you can double check it.

## RAG vs. fine-tuning — how to choose

- If the problem is **"the model doesn't know this fact"** → use RAG. Facts
  change often (prices, policies) — RAG just needs the file updated, no
  retraining.
- If the problem is **"the model doesn't behave the way I want"** (wrong
  tone, wrong format, doesn't follow instructions) → that's fine-tuning's
  job. RAG can't change *how* a model writes, only *what* it can see.

## The four ways RAG breaks (and how we fixed them here)

1. **Bad cutting** — cut a document in the wrong place and you separate a
   rule from the sentence that explains it. Fix: cut on paragraph breaks,
   not just a fixed character count.
2. **Wrong search results** — the search picks the wrong chunk. In our
   project, this really happened: asking about an "unopened" product
   didn't match text that says "unused condition," because our fallback
   search only matches exact words, not meaning. Fix: combine word-matching
   search (BM25) with meaning-matching search (embeddings) together.
3. **Too much stuffed in** — handing the model 50 chunks instead of 4
   buries the useful part in a pile of noise. Fix: only retrieve a small,
   focused number of chunks.
4. **Model still makes stuff up** — even with the right info in front of
   it, a model can ignore it and guess. Fix: tell it directly, "only answer
   from what I gave you, and say so if it's not enough."

## The real bug we hit building this

We tried to download the real AI model (`sentence-transformers`) to turn
text into numbers, but this practice environment isn't allowed to reach
huggingface.co (the site that hosts it) — kind of like being on office
wifi that blocks certain websites. So the code automatically switches to a
simpler, offline word-counting method (TF-IDF) instead, just so we could
keep testing everything else. It's not as smart (it only matches exact
words, not meanings), but it let the whole project run end-to-end without
internet access. On your own computer, which does have normal internet,
the real smarter version will just work.

## The four ways we grade a RAG answer (RAGAS)

- **Faithfulness** — did the answer only say things the documents actually
  support, or did it make something up?
- **Answer Relevance** — did the answer actually address the question
  asked, not just say true-but-unrelated things?
- **Context Precision** — out of what we retrieved, how much of it was
  actually useful (not junk)?
- **Context Recall** — did we retrieve *everything* needed to fully answer,
  or did we miss a piece?

You need all four together because a system can look great on one and be
quietly broken on another — for example, a legal RAG tool scored great on
"faithfulness" for weeks while it was actually missing a key document every
time, because faithfulness only checks "does the answer match what it
found," not "did it find everything it needed to find."
