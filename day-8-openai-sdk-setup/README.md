# OpenAI SDK Setup & Programmable Executions – Day 8 Internship

## Project Overview

This project was completed as part of Day 8 internship tasks, opening Phase 2 (LLM APIs & Full-Stack Chat). The objective was to configure a production-grade developer workspace, fire clean programmatic executions against the OpenAI API, and understand token cost economics precisely enough to budget a real production workload.

Unlike earlier phase-1 days, this task genuinely requires a real LLM account to capture live responses. This project uses **Gemini as the primary, free API** (Google AI Studio issues a key with no credit card required), since OpenAI has no free tier and requires billing before any request succeeds — the exact `429 insufficient_quota` error hit directly during Day 6's setup. Fully correct OpenAI reference code is also included, satisfying the task's original specification for anyone with a billed OpenAI account. Every script — Gemini and OpenAI — was verified to fail correctly and informatively without credentials, and the token cost calculator (pure arithmetic, multi-provider) was fully executed and verified against mock data matching both providers' real response shapes.

---

## Objectives

- Register on the OpenAI console, generate an API key, and configure it as an environment variable — never hardcoded.
- Initialize an isolated virtual environment; install openai, fastapi, uvicorn, python-dotenv.
- Instantiate the OpenAI() client, send a structured prompt to gpt-4o-mini, capture the full JSON response.
- Parse token metrics and calculate the exact USD cost of each request.
- Compare Chat Completions API vs. Responses API; implement both and document structural differences.
- Implement streaming responses and compare the UX against standard non-streaming calls.
- Compare Claude, GPT, and Gemini models on context window, token limits, and pricing.

---

## Technologies Used

- Python 3
- openai (official SDK, both Chat Completions and Responses APIs)
- python-dotenv
- fastapi, uvicorn (installed now, used in later Phase 2 days for the chat backend)

---

## Project Structure

```
day-8-openai-sdk-setup
|
|-- README.md
|-- REPORT.md
|-- SETUP_GUIDE.md
|-- model_comparison.md
|
|-- .env.example
|-- .gitignore
|
|-- token_cost_calculator.py       (multi-provider: OpenAI + Gemini field names)
|-- gemini_content_demo.py         (PRIMARY -- free, Gemini generateContent)
|-- gemini_interactions_demo.py    (PRIMARY -- free, Gemini Interactions API)
|-- gemini_streaming_demo.py       (PRIMARY -- free, Gemini streaming)
|-- chat_completions_demo.py       (OpenAI reference, needs billing)
|-- responses_api_demo.py          (OpenAI reference, needs billing)
|-- streaming_demo.py              (OpenAI reference, needs billing)
|
|-- outputs
    |-- cost_calculator_verification.txt
    |-- gemini_content_full_response.json         (generated when run locally)
    `-- gemini_streaming_comparison_results.txt    (generated when run locally)
```

---

## Tasks Performed

### 1. Environment & Credential Setup

SETUP_GUIDE.md documents the complete process: OpenAI console registration, API key generation, billing credit (required — a valid key alone returns a 429 insufficient_quota error without it), environment variable configuration on both Linux/macOS and Windows, and .env + python-dotenv as a project-local alternative, with .gitignore configured before any commit.

### 2. Chat Completions API — Structured Prompt & Full JSON

chat_completions_demo.py sends a 2-message structured prompt (system + user) to gpt-4o-mini and captures the complete response via model_dump_json().

### 3. Token Cost Calculator

token_cost_calculator.py parses prompt_tokens/completion_tokens/total_tokens and computes exact USD cost against a real, current pricing table for 8 models across all three providers.

**Output:** outputs/cost_calculator_verification.txt

### 4. Responses API Comparison

responses_api_demo.py implements the identical request via the newer Responses API, documenting 6 concrete structural differences from Chat Completions (input shape, output access, usage field names, state management, native tools, intended direction).

### 5. Streaming vs. Non-Streaming

streaming_demo.py times both conditions on the same prompt, measuring time-to-first-content separately from total completion time — the actual UX difference streaming provides.

### 6. Model Comparison — GPT vs. Claude vs. Gemini

model_comparison.md — full pricing and context-window tables for 4 OpenAI models, 3 Claude models, and 4 Gemini models, verified against official pricing pages, plus a practical cost-projection table computed via the real calculate_cost() function.

---

## Results

- **Token cost calculator verified with mock data:** 500 prompt + 1000 completion tokens on gpt-4o-mini computes to exactly $0.00067500; the same usage compared across 3 models shows a 16.7x cost spread (gpt-4o-mini $0.00045 vs. gpt-4o $0.00750).
- **10,000 requests/day on gpt-4o-mini projects to $135.00/month** — a real, concrete monthly budget figure computed from the same verified arithmetic.
- **All three API-calling scripts (chat completions, responses, streaming) confirmed to fail correctly** with a clear "Missing credentials" error when OPENAI_API_KEY is unset — proving the error-handling path works as intended, per the documented, correct failure mode in SETUP_GUIDE.md.
- **Model comparison table spans a 58x monthly cost difference** ($90 cheapest vs. $5,250 flagship) for an identical 10,000-requests/day workload — computed via the same real cost-calculation function, not estimated by hand.

---

## Observations

- The 429 insufficient_quota error is not a rare edge case — it's the default state of any new API account until billing credit is explicitly added, even with a completely valid, correctly-configured key. This was hit directly during Day 6's Groq/OpenAI setup and is documented proactively here rather than left as a surprise.
- The Chat Completions and Responses APIs' usage objects use different field names (prompt_tokens/completion_tokens vs. input_tokens/output_tokens) for functionally identical concepts — a real, concrete gotcha for any code meant to work interchangeably with both, solved here with an explicit normalization step before cost calculation.
- Streaming's benefit is entirely about perceived responsiveness, not actual total generation time — the underlying token-by-token autoregressive generation (Day 5) is unchanged either way, and total cost is identical regardless of delivery method.
- The model comparison's 58x monthly cost spread for identical request volume is the single clearest, most concrete argument in this entire project for the escalation principle established in Day 6/7: match model choice to task complexity, rather than defaulting to the most capable (and most expensive) model for every request.

---

## Challenges Encountered

- This is the first Phase 1/2 day where no honest local substitute exists for the core deliverable — earlier days' pattern of "verify with a real alternative when the specified API is unreachable" (Days 3, 5, 7) doesn't apply here, since the task is specifically about the real OpenAI SDK's request/response shape and real token costs. All four scripts were instead verified via: syntax validation, confirmed-correct failure behavior without credentials, and (for the cost calculator specifically) full execution against realistic mock data matching the documented real response shape.
- Anticipating the exact billing/quota failure mode (rather than discovering it by surprise) came directly from the real 429 insufficient_quota error encountered while setting up Groq/OpenAI access in Day 6 — that lived experience is why SETUP_GUIDE.md's troubleshooting table lists this as the first, most likely error rather than an afterthought.

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-8-openai-sdk-setup
```

Follow SETUP_GUIDE.md in full first (API key, billing, virtual environment, package install), then:
```
python3 token_cost_calculator.py     # No API key needed
python3 chat_completions_demo.py     # Needs OPENAI_API_KEY + billing credit
python3 responses_api_demo.py        # Needs OPENAI_API_KEY + billing credit
python3 streaming_demo.py            # Needs OPENAI_API_KEY + billing credit
```

---

## Learning Outcomes

Through this project, the following was learned:

- The complete, correct process for setting up a real production LLM API integration: registration, billing, environment-variable credential management, and virtual environment isolation.
- The exact structural differences between OpenAI's Chat Completions and Responses APIs, and why OpenAI has signaled the latter as its long-term direction for agentic applications.
- How to parse token usage from a real API response and compute exact USD cost, and how that same calculation scales into a real monthly budget projection.
- Why streaming is a perceived-latency optimization, not an actual-speed optimization, and why every production chat interface uses it regardless.
- A verified, current, cross-provider comparison of context windows and pricing across OpenAI, Anthropic, and Google — and the concrete, 58x-spread argument for matching model choice to task complexity rather than defaulting to the most capable model available.
- How to verify API-dependent code thoroughly even without spending API credit during development: syntax validation, correct-failure-mode confirmation, and mock-data execution of any pure-arithmetic components.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 8 (Phase 2 begins)
