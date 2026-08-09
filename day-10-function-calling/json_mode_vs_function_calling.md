# Function Calling vs. Regular JSON Mode — When to Use Each

## The Core Distinction

Both mechanisms make an LLM produce structured, machine-readable output — but they solve genuinely different problems.

**JSON mode / schema-constrained output (Day 9's territory)** forces the model's final text response to conform to a schema. The model reads your prompt, reasons about it using only its own training knowledge and whatever context you provided, and produces one structured answer. Nothing external happens — no code runs, no database is queried, no real-world action occurs.

**Function calling (Day 10)** lets the model request that your code do something on its behalf, see the real result of that action, and then produce its final response informed by that real result. The model doesn't just format an answer — it participates in a loop with your actual running program.

## The Test That Distinguishes Them

Ask: does answering this question require information or computation the model doesn't already have, and can't reliably produce on its own?

- If no — the model has everything it needs in its training knowledge or the provided context, and just needs to be organized into a specific shape — use JSON mode.
- If yes — the answer depends on live data, precise calculation, or an action in an external system — use function calling.

## Concrete Examples From This Project

| Task | Mechanism | Why |
|---|---|---|
| Extracting {sentiment, confidence, key_points} from a review (Day 9) | JSON mode | The model already has everything it needs to judge sentiment — it just needs to report it in a fixed shape. No external lookup required. |
| "What time is it in Tokyo right now?" | Function calling | The model's training data has no way to know the current time — this requires a real system clock call, not reasoning. |
| Generating a product catalog entry from a description (Day 9) | JSON mode | Purely a formatting/generation task from information already in the prompt. |
| "Is the wireless keyboard in stock, and how much is it?" | Function calling | This is your company's live database — the model cannot possibly know this from training data; it must ask your system. |
| "What's 4127 x 8912 / 3?" | Function calling | Day 6's CoT experiments measured directly that LLMs are unreliable at large-number mental arithmetic (12.5%-50% direct-answer accuracy) — always offload this to a real calculator, never trust the model's own computation for anything that matters. |
| Parsing a support email into {customer_name, issue_category, urgency} (Day 9) | JSON mode | Everything needed is already present in the email text itself. |

## Why You Can't Just Use Function Calling for Everything

It's tempting to think "function calling is strictly more powerful, so just always use it." Two real reasons not to:

1. **Latency and cost.** Function calling requires at least two model round-trips (Part 2's 5-stage loop: send tools -> model decides -> execute -> send result -> model synthesizes) — every one of those round-trips costs real tokens and real time (Day 8's territory). A pure JSON-mode call is one round-trip. If the model already has everything it needs, the extra round-trip is pure waste.

2. **Unnecessary complexity and failure surface.** Every tool call is a new opportunity for the three edge cases demonstrated in this project (model declines incorrectly, invalid arguments, function errors) to occur. A task that doesn't need external data doesn't need to carry that risk.

## Why You Can't Just Use JSON Mode for Everything Either

The reverse mistake is worse: asking a model to produce structured output about something it genuinely cannot know (today's date, your company's live inventory, an exact large-number calculation) via JSON mode alone will produce confident, well-formatted, wrong answers — the schema constrains the shape of the hallucination, not whether it's a hallucination. This is precisely the RAG lesson from Day 7 generalized: grounding matters more than formatting. Function calling is how you ground a response in real, current, verifiable data; JSON mode alone cannot do this.

## The Combined Pattern (What Real Production Systems Actually Do)

Most real systems use both, together, in sequence: function calling to gather real, current, or precise information, followed by a JSON-mode-style final response schema so that even the synthesized answer comes back in a predictable shape your application can parse. multi_tool_agent.py in this project demonstrates the function-calling half; combining its final answer with Day 9's schema-enforcement pattern would complete the full production pattern.

## Summary Table

| | JSON Mode | Function Calling |
|---|---|---|
| Model round-trips | 1 | 2+ |
| Can access external/live data | No | Yes |
| Can perform precise calculation | No -- relies on model's own (unreliable) arithmetic | Yes -- delegates to real code |
| Risk of hallucinated-but-well-formatted answers | Higher, for anything outside training knowledge | Lower -- grounded in real tool results |
| Added latency/cost | Minimal | Real, measurable (Day 8) |
| Added failure surface | Schema validation only (Day 9) | Schema validation + 3 more edge case categories (Day 10) |
| Best for | Formatting/extracting from already-available information | Live data, precise computation, real-world actions |
