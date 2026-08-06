# Model Comparison: GPT vs. Claude vs. Gemini

**Pricing verified against official provider pricing pages and cross-checked against independent trackers, current as of August 2026.** All providers revise pricing periodically -- always confirm against the live pricing page before budgeting a production workload.

---

## OpenAI (GPT)

| Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Notes |
|---|---|---|---|---|
| GPT-4o mini | 128,000 tokens | $0.15 | $0.60 | Cheapest OpenAI model in general use; used throughout this project |
| GPT-4o | 128,000 tokens | $2.50 | $10.00 | Flagship multimodal model; ~16x more expensive than 4o mini |
| GPT-4.1 mini | 1,047,576 tokens | $0.40 | $1.60 | Massive context jump over 4o-mini/4o at a modest price increase |
| GPT-5 (reasoning) | ~400,000 tokens | ~$1.25 | ~$10.00 | Reasoning-tier model; extended "thinking" tokens add real per-request cost |

---

## Anthropic (Claude)

| Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Notes |
|---|---|---|---|---|
| Claude Haiku 4.5 | 200,000 tokens | $1.00 | $5.00 | Fastest, cheapest current Claude tier |
| Claude Sonnet 5 | 1,000,000 tokens (128K max output) | $2.00* | $10.00* | *Introductory pricing through Aug 31, 2026; standard $3/$15 from Sep 1, 2026 |
| Claude Opus 5 | 1,000,000 tokens (128K max output) | $5.00 | $25.00 | Flagship reasoning/agentic-coding tier |

**Universal Claude pattern:** output is priced at roughly 5x input across the current lineup -- a useful, simple budgeting heuristic. Prompt caching cuts cached-input cost by ~90%; batch processing cuts both input and output by 50%.

---

## Google (Gemini)

| Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Notes |
|---|---|---|---|---|
| Gemini 2.5 Flash-Lite | 1,000,000 tokens | $0.10 | $0.40 | Cheapest model across all three providers |
| Gemini 2.5 Flash | 1,000,000 tokens | $0.30 | $2.50 | Cost-effective general workhorse |
| Gemini 2.5 Pro | 1,000,000 tokens (2x rate above 200K) | $1.25 (<=200K) / $2.50 (>200K) | $10.00 (<=200K) / $15.00 (>200K) | Tiered pricing -- cost roughly doubles for prompts over 200K tokens |
| Gemini 3.1 Pro (preview) | 2,000,000 tokens | $2.00 (<=200K) | $12.00 (<=200K) | Largest context window of any model compared here |

**Gemini's distinctive pattern:** every tier -- even the cheapest, Flash-Lite -- gets a 1M-token context window. Neither OpenAI nor Anthropic offers 1M context at their cheapest price point; Gemini does.

---

## Side-by-Side: Cheapest Tier vs. Flagship Tier

| | Cheapest tier | Flagship tier |
|---|---|---|
| **OpenAI** | GPT-4o mini: $0.15/$0.60, 128K context | GPT-5: ~$1.25/$10, ~400K context |
| **Claude** | Haiku 4.5: $1.00/$5.00, 200K context | Opus 5: $5.00/$25.00, 1M context |
| **Gemini** | Flash-Lite: $0.10/$0.40, 1M context | 3.1 Pro: $2.00/$12.00 (<=200K), 2M context |

**Key takeaways:**

1. **Gemini is the cheapest and most context-generous at the low end** -- Flash-Lite beats every other cheapest-tier model on price while matching or beating them on context window.
2. **OpenAI's GPT-4o mini remains extremely competitive for general-purpose, moderate-context tasks** -- its 128K window is the smallest of the three at the budget tier, but the price is close to Gemini's cheapest option.
3. **Claude has no true "ultra-budget" tier** -- Haiku 4.5 at $1/$5 is meaningfully more expensive than GPT-4o mini or Gemini Flash-Lite, but Claude's lineup is priced with a clean, predictable 5x input-to-output ratio across every tier, simplifying budgeting.
4. **Context window size is no longer a flagship-only feature** -- all three providers now offer 1M-token context even at moderate price points; Gemini 3.1 Pro's 2M window is currently the largest available anywhere in this comparison.
5. **Output tokens are always more expensive than input tokens**, typically 4-8x, across every provider and every tier -- a real, practical implication: verbose response styles (e.g. Chain-of-Thought reasoning, Day 6) cost disproportionately more than the input prompt driving them.

---

## Practical Cost Example: 10,000 Requests/Day (1,000 input + 500 output tokens each)

*(All figures below were computed and verified directly using `token_cost_calculator.py`'s real arithmetic — not estimated by hand.)*

| Model | Cost/Request | Daily Cost | Monthly Cost (x30) |
|---|---|---|---|
| Gemini 2.5 Flash-Lite | $0.00030 | $3.00 | $90.00 |
| GPT-4o mini | $0.00045 | $4.50 | $135.00 |
| GPT-4.1 mini | $0.00120 | $12.00 | $360.00 |
| Claude Haiku 4.5 | $0.00350 | $35.00 | $1,050.00 |
| Gemini 2.5 Pro | $0.00625 | $62.50 | $1,875.00 |
| Claude Sonnet 5 | $0.00700 | $70.00 | $2,100.00 |
| GPT-4o | $0.00750 | $75.00 | $2,250.00 |
| Claude Opus 5 | $0.01750 | $175.00 | $5,250.00 |

**This table is the concrete argument for the escalation principle covered in Day 6/7:** the cheapest tier (Gemini Flash-Lite, $90/month) versus the most expensive flagship (Claude Opus 5, $5,250/month) is a **58x difference in monthly spend** for the exact same request volume and shape. For high-volume, moderate-complexity production workloads, the cheapest model that clears your quality bar matters enormously — reflexively defaulting to a flagship model for every request is one of the most common, avoidable cost mistakes in production LLM systems.
