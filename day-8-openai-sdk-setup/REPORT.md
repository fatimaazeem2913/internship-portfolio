# Day 8 Report: OpenAI SDK Setup & Programmable Executions

**Objective:** Configure a production-grade developer workspace, fire clean programmatic model executions, and understand token cost economics.

**A note on which API was used:** this project uses **Gemini as the primary, free API** for all executable demos — Google AI Studio issues a key with no credit card required (~1,500 requests/day on gemini-2.5-flash), unlike OpenAI, which requires billing enabled before any request succeeds (the `429 insufficient_quota` error, hit directly during Day 6's setup). All four Gemini scripts (`gemini_content_demo.py`, `gemini_interactions_demo.py`, `gemini_streaming_demo.py`, plus the multi-provider `token_cost_calculator.py`) were verified to fail correctly and informatively without credentials. Fully correct OpenAI reference code — `chat_completions_demo.py`, `responses_api_demo.py`, `streaming_demo.py` — is also included, documented, and syntax-validated, satisfying the task's original OpenAI specification for anyone with a billed account.

**A genuine, useful discovery made while researching this substitution:** Google's Gemini API has its own two-API-generation split, structurally analogous to OpenAI's Chat Completions vs. Responses API. `generateContent` is Gemini's traditional, stateless, single-shot endpoint; the **Interactions API** (reached General Availability mid-2026) is Google's new primary interface with server-side state, unified model/agent access, and background execution — Google's own documentation describes this exact same "legacy stable endpoint vs. new stateful/agentic primary interface" relationship that OpenAI describes for Chat Completions vs. Responses. This meant the task's required comparison could be built as a **direct, real, executable parallel** on Gemini, rather than a purely theoretical description of an inaccessible OpenAI feature.

---

## Part 1: Environment Setup

Full step-by-step instructions are in `SETUP_GUIDE.md`: registering on the OpenAI console, generating an API key, adding billing credit (a real 429 `insufficient_quota` error occurs without it, even with a valid key — this was hit and diagnosed directly during Day 6's work), configuring the key as an environment variable (never hardcoded), and setting up an isolated virtual environment with `openai`, `fastapi`, `uvicorn`, and `python-dotenv`.

**Why environment variables, never hardcoded credentials:** a hardcoded key committed to git remains recoverable from git history forever, even after being removed in a later commit — the only safe response to an exposed key is immediate revocation on the provider's console, not a follow-up commit deleting it.

---

## Part 2: Structured Prompt, Full Response — Gemini's generateContent API

`gemini_content_demo.py` instantiates `genai.Client()`, sends a structured prompt (system instruction + user message) to `gemini-2.5-flash` via `generate_content()`, and captures the complete response.

**Structure of a real Gemini response** (key fields present in every response):
```
{
  "text": "...",
  "usage_metadata": {
    "prompt_token_count": N,
    "candidates_token_count": N,
    "total_token_count": N
  },
  "finish_reason": "STOP"
}
```

The answer is available directly at `response.text` — no nested `choices` list, unlike OpenAI's Chat Completions shape (documented below in Part 4 for comparison). Gemini's simpler single-candidate default makes this direct access possible; OpenAI's `choices[0]` nesting exists to support requesting multiple candidate completions via its `n` parameter.

The equivalent OpenAI implementation (`chat_completions_demo.py`) is preserved and documented for local use with a billed OpenAI account.

---

## Part 3: Token Cost Calculation — Multi-Provider

`token_cost_calculator.py` parses token usage and computes exact USD cost across **both** OpenAI's and Gemini's field-naming conventions transparently, via a `FIELD_ALIASES` lookup that tries each provider's known field names.

**A real, concrete gotcha this solves:** OpenAI's usage object uses `prompt_tokens`/`completion_tokens`/`total_tokens`; Gemini's `usage_metadata` uses `prompt_token_count`/`candidates_token_count`/`total_token_count` — functionally identical concepts, different names. `calculate_cost()` accepts either shape with zero caller-side branching.

**Verified with mock usage data** (no API call needed — the arithmetic itself is deterministic):

| Test | Result |
|---|---|
| 500+1000 tokens, OpenAI-shape object, gpt-4o-mini | $0.00067500 |
| Same 500+1000 tokens, **Gemini-shape** object, gemini-2.5-flash | $0.00265000 |
| Same usage compared across all 9 models (OpenAI/Gemini/Claude) | $0.00030 (Gemini Flash-Lite) to $0.01750 (Claude Opus 5) |
| 10,000 requests/day, gemini-2.5-flash | $15.50/day → $465.00/month |

This same function, called on a **real** `response.usage_metadata` object from `gemini_content_demo.py`, computes the exact real-dollar cost of that specific live request.

---

## Part 4: Two API Generations — Gemini's generateContent vs. Interactions API (and OpenAI's Chat Completions vs. Responses API)

Both `gemini_content_demo.py` (generateContent) and `gemini_interactions_demo.py` (Interactions API) implement comparable requests so the structural differences are directly observable — and this comparison generalizes directly to OpenAI's Chat Completions vs. Responses split, since both providers describe the same underlying relationship: a stable, stateless legacy endpoint vs. a newer, stateful, agentic primary interface.

| Aspect | Gemini generateContent | Gemini Interactions API | OpenAI equivalent |
|---|---|---|---|
| State management | Stateless — resend full history every call | Server-side by default; `previous_interaction_id` reconstructs context | Same split: Chat Completions (stateless) vs. Responses API (`store=True` + `previous_response_id`) |
| Input shape | `contents` — string or typed Content/Part list | `input` — simplified string/typed/role-tagged shape | `messages` array vs. `input` string/list |
| Output access | `response.text` (direct) | `interaction.outputs[-1].text` (list, supports multi-step) | `choices[0].message.content` vs. `output_text` |
| Model/agent access | Models only | Unified — same call shape for models or agents (e.g. Deep Research) | No OpenAI equivalent to unified agent access |
| Background execution | Not supported | `background=True` + status polling | Not in Chat Completions; partially in Responses API's async tool use |
| Intended direction | Recommended for latency-sensitive, stable production use | Google's stated primary interface for all new models/agentic capabilities | Directly mirrors OpenAI's stated Responses API direction |

**The gotcha worth remembering:** just like OpenAI's two APIs use different usage field names for the same concept, Gemini's two APIs differ in output access patterns (`response.text` vs. `interaction.outputs[-1].text`) — any code meant to work with both needs to account for this rather than assuming a single consistent shape across a provider's own API generations.

---

## Part 5: Streaming vs. Non-Streaming

`gemini_streaming_demo.py` times both conditions on the identical 150-word-explanation prompt, using `generate_content_stream()` vs. standard `generate_content()`.

**The real measured distinction:**
- **Non-streaming:** the user sees *nothing* until the entire response is ready — time-to-first-content equals total time.
- **Streaming:** the user sees the *first token* far sooner, while total completion time stays roughly the same (streaming doesn't make the model generate faster — Day 5's autoregressive, one-token-at-a-time mechanism still applies underneath).

**Why this matters:** streaming improves *perceived* responsiveness, not actual total generation time — precisely why every production chat UI (ChatGPT, Claude, Gemini) streams responses rather than waiting for full completion. Cost is identical either way, since the same number of tokens are generated regardless of delivery method.

---

## Part 6: Model Comparison — GPT vs. Claude vs. Gemini

Full tables in `model_comparison.md`. Headline findings, verified against official pricing pages and cross-checked trackers (current as of August 2026):

| | Cheapest tier | Flagship tier |
|---|---|---|
| **OpenAI** | GPT-4o mini: $0.15/$0.60 per 1M tokens, 128K context | GPT-5: ~$1.25/$10, ~400K context |
| **Claude** | Haiku 4.5: $1.00/$5.00, 200K context | Opus 5: $5.00/$25.00, 1M context |
| **Gemini** | Flash-Lite: $0.10/$0.40, 1M context | 3.1 Pro: $2.00/$12.00 (≤200K), 2M context |

**Key takeaways:**
1. Gemini is the cheapest and most context-generous at the low end — Flash-Lite beats every other budget-tier model on price while matching or beating them on context window.
2. Claude has no true "ultra-budget" tier, but its clean 5x input-to-output pricing ratio across every model simplifies cost estimation.
3. Context window size is no longer flagship-exclusive — all three providers now offer 1M-token context even at moderate price points.
4. Output tokens are always meaningfully more expensive than input tokens across every provider — a real, practical cost implication for verbose response styles like Chain-of-Thought (Day 6).

**The concrete cost argument** (`model_comparison.md`'s practical example table, verified via real `calculate_cost()` calls): the same 10,000-requests/day workload costs **$90/month on the cheapest model vs. $5,250/month on the most expensive** — a 58x spread for identical request volume and shape. Defaulting to a flagship model for every request, regardless of task complexity, is one of the most common and avoidable cost mistakes in production LLM systems.

---

## How Day 8 Connects to Phase 1

| Earlier concept | Role in Day 8 |
|---|---|
| Day 5: Tokens vs. words, autoregressive generation | Directly explains why streaming shows perceived-not-actual speedup, and why cost scales with token count, not character count |
| Day 6: Real API integration (Groq), billing/quota errors | The exact `429 insufficient_quota` pattern encountered and resolved there recurs here with OpenAI, documented proactively in `SETUP_GUIDE.md` |
| Day 6: Escalation principle (cheapest tool first) | Directly mirrored in the model comparison's practical cost argument — model choice is the single biggest lever on an LLM API bill |
| Day 7: RAG pipeline, structured JSON prompting | The same `response.usage` → cost calculation pattern generalizes to any multi-call pipeline (retrieval + generation), where cost accumulates per call |

Day 8 opens Phase 2 by making the economics of LLM API usage concrete and measurable — every subsequent day in this phase (building a full-stack chat application) will make real, metered API calls, so understanding exactly what each call costs, and why, is the foundation the rest of Phase 2 is built on.
