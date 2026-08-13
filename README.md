# AI/ML Internship Portfolio

This repository contains all work completed during my AI/ML internship, organized by day. Each day builds on the previous one, tracing the progression from classical NLP through modern context-sensitive representations.

## Structure

| Day | Focus | Folder |
|---|---|---|
| Day 1 | Classical NLP preprocessing pipeline — text cleaning, tokenization comparison, stemming/lemmatization, POS tagging, Bag-of-Words | [`day-1-nlp-pipeline/`](./day-1-nlp-pipeline) |
| Day 2 | From counting to predicting — TF-IDF manual verification, cosine retrieval, synonym-blindness proof, Word2Vec skip-gram, polysemy failure demonstration | [`day-2-tfidf-word2vec/`](./day-2-tfidf-word2vec) |
| Day 3 | From static vectors to context — N-gram language models, LSTM gradient analysis, BERT contextual embeddings, t-SNE visualization | [`day-3-context-attention/`](./day-3-context-attention) |
| Day 4 | Transformer architecture — attention from scratch in NumPy, positional encoding, full block implementation, GPT-2 decoder-only study | [`day-4-transformer-architecture/`](./day-4-transformer-architecture) |
| Day 5 | How LLMs work internally — training pipeline, autoregressive generation, sampling strategies from scratch, architecture comparison, emergent capabilities | [`day-5-llm-internals/`](./day-5-llm-internals) |
| Day 6 | Prompt engineering fundamentals — prompt anatomy, zero/one/few-shot comparison, Chain-of-Thought accuracy testing, ReAct pattern, reusable template library | [`day-6-prompt-engineering/`](./day-6-prompt-engineering) |
| Day 7 | Week 1 review mini-project — full RAG pipeline (PDF extraction, 3-way retrieval, structured LLM prompting), code refactoring, and a complete Transformer walkthrough | [`day-7-week1-review/`](./day-7-week1-review) |
| Day 8 | OpenAI SDK setup & programmable executions — structured prompts, token cost economics, Chat Completions vs Responses API (via Gemini's parallel generateContent vs Interactions API), streaming, and a cross-provider model comparison | [`day-8-openai-sdk-setup/`](./day-8-openai-sdk-setup) |
| Day 9 | System personas, roles & structured output — role separation, dual-layer JSON schema validation, few-shot payloads, 4 production prompt types, and a real, live-discovered Gemini 3.x temperature deprecation finding, verified with genuine test data | [`day-9-personas-structured-output/`](./day-9-personas-structured-output) |
| Day 10 | Function calling & tool use — 4 custom tools with JSON schemas, the full 5-stage function calling loop, a multi-tool agent chaining 4 calls, all 3 required edge cases verified (including a real bug found and fixed), and JSON mode vs. function calling documented | [`day-10-function-calling/`](./day-10-function-calling) |
| Day 11 | FastAPI backend & chat state management — POST /api/chat and GET /api/sessions with Pydantic schemas, in-memory session store, all required status codes (400/404/422/500) and structured logging, verified with 10/10 real tests against the running application | [`day-11-fastapi-backend/`](./day-11-fastapi-backend) |
| Day 12 | React chat interface — Vite + Tailwind CSS v4, distinct user/assistant message styling, submit-on-Enter input, CORS verified with real header inspection tests, graceful fetch() error handling, typing indicator, professional layout | [`day-12-react-chat-interface/`](./day-12-react-chat-interface) |
| Day 13 | Multi-session chat, sidebar & advanced UX — sidebar with session switching, crypto.randomUUID() New Chat, AI-generated titles, copy/regenerate actions, localStorage persistence, and server-side session isolation verified via a direct 5-session adversarial test (0 cross-contamination) | [`day-13-multisession-chat/`](./day-13-multisession-chat) |
## The Full Arc

Each day is designed to expose a specific limitation of the previous day's approach:

- **Day 1** builds classical, purely lexical representations that treat text as strings — no meaning captured.
- **Day 2** replaces counting with predictive training (Word2Vec), demonstrably improving synonym detection but still producing one fixed vector per word — proven to fail on polysemy.
- **Day 3** introduces sequential models (n-grams, LSTMs), measures the vanishing-gradient problem directly, and uses pretrained BERT to resolve Day 2's open question by producing genuinely context-dependent embeddings.
- **Day 4** builds the Transformer from first principles in pure NumPy, showing how one architecture resolves every limitation measured in Days 1–3 — and why the O(n²) cost of attention became the defining constraint of the LLM era.
The full technical progression is documented in each day's `REPORT.md` and `README.md`.
- **Day 5** shows how these components are assembled into a real training pipeline (pre-training → SFT → RLHF), demonstrates that autoregressive generation is just "predict, sample, append, repeat" using a real self-built language model, and researches the emergent capabilities that appear only once this simple objective is scaled far enough.
- **Day 6** shows how to get reliable, controllable behavior out of a model whose weights you cannot change — measuring, against a real, independent production model (Llama 3.3 70B via Groq's free API), exactly how much structure, demonstration, and explicit reasoning improve output quality and accuracy (CoT: 50% → 87.5%), building the tool-use pattern (ReAct) that turns a language model into an agent, and along the way finding and fixing a real tool-matching bug and documenting a genuine few-shot-induced hallucination — the kind of finding only real API testing surfaces.
- **Day 7** assembles everything from Week 1 into one working system — a real retrieval-augmented generation pipeline — and closes with a full stage-by-stage Transformer walkthrough tracing a real query from raw text to a sampled output token.
## Phase 2: LLM APIs & Full-Stack Chat (Days 8-14)

- **Day 8** opens Phase 2 by making LLM API economics concrete — real token cost calculation across OpenAI, Gemini, and Claude, a genuine architectural parallel discovered between Gemini's Interactions API and OpenAI's Responses API, and a verified 58x monthly cost spread across provider tiers for identical request volume.

- **Day 9** turns Day 6's prompt-engineering principles into tested production code, and along the way discovers and proves — with real executed data — that Google silently deprecated temperature control on its newest Gemini models: a creative task produced 3 fully distinct outputs even at temperature=0.0, direct empirical evidence the parameter no longer does anything.

- **Day 10** turns Day 6's prompted ReAct pattern into native, schema-enforced function calling — 4 real tools, a genuine multi-call agent, and a real bug (format_currency crashing on a wrong-typed argument) found through actual testing and fixed with explicit runtime validation.

- **Day 11** exposes everything built in Phase 2 so far over real HTTP — a genuine FastAPI server with server-side session management solving Day 4's "a Transformer has no memory" problem at the application layer, verified with 10/10 real tests covering every required status code and a genuinely simulated failure case.
- **Day 12** completes the first full-stack loop of Phase 2 — a real React UI talking over real HTTP, with CORS permission genuinely verified through header inspection, to the FastAPI backend built in Day 11.
- **Day 13** upgrades the chat app to hold multiple genuinely isolated conversations at once — verified with a direct, adversarial test sending 5 unrelated topics to 5 simultaneous sessions and confirming zero cross-contamination, plus AI-generated titles, persistent sessions across refresh, and message-level copy/regenerate actions.
## Author

**Fatima Azeem**
AI/ML Internship

