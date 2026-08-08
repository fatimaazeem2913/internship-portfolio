# Day 9 Report: System Personas, Roles & Structured Output

**Objective:** Control model behavior precisely by separating context across system/user/assistant roles and enforcing structured JSON output formats.

**A note on API used:** every script uses Gemini as the primary, free API (Day 8's established pattern) — Google AI Studio issues a key with no credit card required. All six scripts were verified to fail correctly and informatively without credentials, and the JSON validation logic (pure Python, no API dependency) was fully executed against 7 test cases, both valid and deliberately broken.

---

## Part 1: System / User / Assistant Role Separation

`roles_and_messages_demo.py` builds a genuine multi-turn conversation using all three roles: a persistent `system_instruction` (never resent, applies throughout), a `user`/`model` turn history (Gemini's naming for user/assistant), and a final new user question.

**Why three roles instead of one prompt:** mixing instructions and conversation into a single blob forces the model to guess which parts are permanent rules versus one-time input. Separating them lets the system role persist unchanged as a behavioral contract, while user/assistant turns carry the actual back-and-forth.

**The verification test built into the script:** the conversation first establishes that the app has offline mode, then asks about Pro plan pricing — while the system instruction explicitly forbids inventing pricing information. A correct response must simultaneously (1) decline to state a specific price, honoring the system role, and (2) respond as a coherent continuation of the offline-mode exchange, honoring the assistant-role history — proving both roles are genuinely being used together, not just present as unused decoration.

---

## Part 2: Strict JSON Schema Enforcement — Real, Verified Validation Logic

`json_schema_enforcement.py` implements two layers of defense:

1. **API-level:** Gemini's `response_json_schema` + `response_mime_type="application/json"` constrains the model's token sampling itself — a stronger guarantee than a prompt instruction alone (Day 6 study guide's structured-output principle).
2. **Code-level:** even with API constraints, a production system must validate independently, since malformed output can still occur.

**The code-level validation was fully executed against 7 real test cases** — 2 valid/recoverable, 5 deliberately broken in distinct, realistic ways:

| Test case | Result |
|---|---|
| Valid, well-formed JSON | OK — parsed and validated |
| Wrapped in markdown code fences (a real, common model deviation) | OK — the parser strips fences before parsing |
| Missing a required field | VALIDATION_ERROR — correctly names the missing field |
| Invalid enum value (model didn't respect allowed sentiment values) | VALIDATION_ERROR — correctly flags the bad value |
| confidence out of the required 0-1 range | VALIDATION_ERROR — correctly flags the range violation |
| Completely malformed JSON | PARSE_ERROR — caught before validation even runs |
| Wrong type (string instead of array) | VALIDATION_ERROR — correctly names the type mismatch |

Every failure mode was caught with a distinct, correct, actionable error message — none of the 5 broken cases crashed the pipeline or were silently accepted as valid. This is the real, defensible meaning of "handle any deviation in code": not just wrapping a call in try/except, but validating structure, types, and value constraints explicitly.

---

## Part 3: Few-Shot Examples Embedded in the Message Payload

`few_shot_payload_demo.py` builds few-shot examples as genuine alternating user/model conversation turns, rather than describing the pattern in prose inside the system instruction.

**Why this matters architecturally:** examples-as-messages are processed by the model using the exact same mechanism as real conversation history (Part 1) — there's no special case the model has to recognize as "instructional text," it's structurally indistinguishable from "this conversation already happened." This is a more direct, literal application of Day 6's "demonstration beats description" principle than embedding the same examples as prose.

**The test task (date normalization to ISO 8601) includes a deliberately ambiguous case:** "03/09/2026" could mean March 9th or September 3rd depending on convention. The few-shot examples include "12/25/2025" -> "2025-12-25" — unambiguous, since no month is numbered 25, which implicitly establishes MM/DD/YYYY as the convention. This tests whether the model actually uses the demonstrated convention on a genuinely ambiguous new case, not just pattern-matches on easy, unambiguous inputs.

---

## Part 4: Four Production Prompt Types

`production_prompts_demo.py` runs all four required prompt types, each loaded from `prompts/production_prompts.md` (kept separate from application code):

1. **Structured JSON generation** — generates a product catalog entry from a plain description; output is verified parseable via the same `safe_parse_json()` from Part 2.
2. **Unstructured text parsing** — extracts customer_name, issue_category, urgency, and summary from a realistic, messy support email with no explicit field labels.
3. **Code generation** — writes a Python running-median function with an explicitly stated edge case (empty stream) that the system prompt requires be handled.
4. **Document summarization** — summarizes a real factual paragraph (James Webb Space Telescope) to an exact sentence count, testing whether specific numbers (launch date, mirror segment count, diameter) survive compression.

---

## Part 5: Temperature Experiments — A Real, Current API-Deprecation Finding

temperature_experiment.py was originally designed exactly like Day 5/6's temperature experiments: run the same prompt 3 times at 3 temperatures, count distinct outputs. Direct testing surfaced that this no longer works as expected on the current Gemini model generation — and the real results prove it cleanly.

What actually happened: the original model (gemini-2.5-flash) returned a live 404 — deprecated for new users as of Google's August 2026 migration to Gemini 3.x. After switching to the current model (gemini-3.5-flash-lite), Google's own migration documentation confirms that temperature, top_p, and top_k are now deprecated and silently ignored on every Gemini 3.x model — the request succeeds normally (HTTP 200), but the sampling behavior no longer changes with the value supplied.

The real, executed results prove this directly:

| Task | Temperature | Unique outputs (3 runs) |
|---|---|---|
| Factual classification | 0.0 | 1/3 |
| Factual classification | 0.7 | 1/3 |
| Factual classification | 1.5 | 1/3 |
| Creative tagline | 0.0 | 3/3 |
| Creative tagline | 0.7 | 3/3 |
| Creative tagline | 1.5 | 3/3 |

The smoking gun is the creative task at temperature=0.0. If temperature genuinely controlled sampling, temperature=0.0 should push the model toward its single most probable output — near-deterministic, with repeated or near-identical results across runs, exactly as the factual task shows at every temperature. Instead, the creative task produced 3 fully distinct taglines even at temperature=0.0 ("Leave the map behind, follow your feet, and conquer the wild," "Leave the trail behind, take the wild with you," "Leave the map behind and let the earth guide you") — the same degree of variation as at temperature=1.5. This is direct, empirical proof that the temperature parameter is not affecting sampling at all on gemini-3.5-flash-lite, exactly matching Google's stated deprecation.

Why this is a genuinely dangerous gotcha, not a minor detail: a production pipeline sending temperature=0 for deterministic classification output would silently stop being deterministic the moment the underlying model was upgraded — with no error to signal the change. This is exactly the kind of failure that's invisible until someone downstream notices inconsistent results and has to trace it back to an unannounced API behavior change.

Google's documented replacement: explicit system-instruction phrasing controls determinism vs. variety instead of sampling parameters, since Gemini 3.x models are reasoning-first models trained around fixed internal sampling configurations that are no longer meant to be user-tunable. A separate parameter, thinking_level (MINIMAL/LOW/MEDIUM/HIGH), controls reasoning depth and cost/latency — a different axis entirely, not a temperature replacement; use minimal for high-volume simple classification and medium/high for complex multi-step reasoning.

Updated production guidance, now split by model generation:

For current Gemini 3.x models specifically: use explicit system-instruction phrasing for determinism/variety control, not temperature — it has no effect, as directly demonstrated above.

For any model still exposing standard sampling parameters (OpenAI, Claude, Groq, or Gemini 2.5-generation models while still available): the original Day 5/6/9 guidance is unchanged — low temperature (0.0-0.2) for classification/extraction/code generation/RAG, medium (0.5-0.8) for general conversation, high (1.0+) for creative work.

The broader lesson, worth stating plainly: model providers change their APIs, sometimes silently, and code that worked correctly against one model generation can quietly stop working as intended against the next. This project's response — discover it through real testing, prove it with real executed data, document it honestly — is the same discipline established in Day 6's ReAct bug-fix: found live, verified with real numbers, documented transparently rather than hidden.
## Part 6: Persona-Switching System

`persona_switcher.py` defines 3 personas (formal, casual, technical) as separate system prompts in `prompts/personas.md`, and answers the identical question ("Why is my website loading slowly?") under each — isolating the persona's effect from the question, so any difference in output is attributable purely to the system role.

**A PersonaSession class demonstrates runtime persona switching** — a single session starts as casual, answers a question, then switches to technical via .switch_to() and answers the same question again, showing that persona-switching is a live, toggleable property of a conversation, not something requiring a new client or a restarted session.

**What should differ between personas:** tone, sentence structure, contraction use, and technical density — NOT the underlying factual correctness of the troubleshooting advice given. All three personas should identify genuinely valid causes of slow page loads; only the voice should change.

---

## Model Comparison: Claude vs. GPT vs. Gemini (Reused, Verified in Day 8)

`model_comparison.md` — carried forward from Day 8, where it was verified against official pricing pages current as of August 2026. Headline finding, still accurate: Gemini Flash-Lite is the cheapest and most context-generous budget option ($0.10/$0.40 per 1M tokens, 1M context); the same 10,000-requests/day workload spans a 58x monthly cost difference across the full provider/tier range ($90 cheapest to $5,250 flagship).

---

## How Day 9 Connects to Earlier Days

| Earlier concept | Role in Day 9 |
|---|---|
| Day 6: Prompt anatomy (Role + Context + Examples + Format) | Every component individually built out as real, working, executable code across Parts 1, 2, 3, 6 |
| Day 6: Structured output / schema-constrained decoding | Directly implemented and validated in Part 2 |
| Day 5/6: Temperature mechanics | Re-verified experimentally in Part 5, now explicitly tied to task-type decision guidance |
| Day 7: RAG should run at low temperature | Reconfirmed as a general principle in Part 5's production guidance |
| Day 8: Gemini as free primary API, token/cost patterns | Every script in this project builds directly on Day 8's client setup and error-handling patterns |

Day 9 is where Day 6's prompt-engineering principles become Day 8's production code — every concept from the prompt anatomy is now a tested, reusable, importable Python component rather than a one-off demonstration.
