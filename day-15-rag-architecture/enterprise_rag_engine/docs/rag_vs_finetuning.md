# RAG vs. Fine-Tuning

Both techniques adapt an LLM to a specific need, but they change *different
things*: RAG changes what the model can **see** at inference time;
fine-tuning changes what the model **is** (its weights). They are not
mutually exclusive, but picking the wrong one for a given problem wastes
time and money.

## 5 Scenarios Where RAG Is the Right Choice

1. **The knowledge changes frequently.**
   A company's refund policy, pricing, or inventory changes weekly.
   Fine-tuning would require re-training every time something changes; RAG
   just needs the corpus file updated and re-indexed — no re-training.

2. **You need traceable, citable answers.**
   In this project's `enterprise_rag_engine`, every answer returns
   `sources: [...]`, naming the exact file the answer came from. A
   fine-tuned model's answer is baked into opaque weights — there is no way
   to point to "this is the sentence in the source document I got this
   from."

3. **The knowledge base is too large to fit in training data economically.**
   A support corpus with thousands of documents can be indexed into a
   vector store cheaply. Fine-tuning a model to *memorize* thousands of
   documents verbatim is both far more expensive and unreliable — models
   are bad at reliably memorizing large volumes of exact facts through
   fine-tuning; they're much better at using facts placed directly in
   context.

4. **You need to prevent hallucination on out-of-scope questions.**
   A well-built RAG system can say "the provided context doesn't cover
   that" (see `llm.py`'s system prompt) when no relevant chunk is
   retrieved. A fine-tuned model has no equivalent hard boundary — it will
   still generate a fluent-sounding guess for questions its fine-tuning
   data didn't cover.

5. **Multiple tenants/customers need different, swappable knowledge.**
   A SaaS support bot serving many customers, each with their own policy
   documents, can reuse one shared LLM and swap in a different vector store
   per customer. Fine-tuning a separate model per customer would multiply
   hosting cost linearly with customer count.

## 3 Scenarios Where Fine-Tuning Wins

1. **You need to change *how* the model behaves, not what it knows.**
   Teaching a model a consistent tone, output format, or reasoning style
   (e.g. always respond in valid JSON matching a schema, always reason
   step-by-step before answering) is a behavioral change baked into
   weights. RAG cannot make a model *behave* differently — it can only
   change what facts are available to it.

2. **The domain requires deep, implicit pattern learning, not lookup.**
   Teaching a model to write code in a company's specific internal style
   conventions, or to recognize domain-specific implicit patterns (e.g.
   medical note shorthand), is closer to teaching a skill than teaching a
   fact. That's a weights-level change fine-tuning is suited for.

3. **Latency and cost matter more than freshness, and the knowledge is
   genuinely static.**
   Retrieval adds a search step (embedding the query + vector lookup)
   before every generation. If the knowledge truly never changes and
   ultra-low latency matters more than the ability to update facts on the
   fly, a fine-tuned model skips that retrieval step entirely.

## The honest middle ground

In practice, production systems increasingly combine both: fine-tune a
model to reliably follow a RAG-specific *format* (cite sources, refuse
when context is insufficient, output structured JSON), while RAG continues
to supply the actual facts. Neither technique alone is "better" in the
abstract — the right choice depends on whether the problem is "the model
doesn't know this fact" (RAG) or "the model doesn't behave this way" (fine-
tuning).
