# Day 10 Report: Function Calling & Tool Use

**Objective:** Master function calling to build LLMs that interact with external systems — the core mechanism behind AI agents and tool-augmented applications.

**A note on the API used:** every script uses Gemini (`gemini-3.5-flash-lite`, current as of Day 9's real deprecation finding) as the primary, free API. Gemini's function calling mechanism is structurally equivalent to OpenAI's — a `tools` array of JSON-schema function declarations, a `tool_config.function_calling_config.mode` parameter directly analogous to OpenAI's `tool_choice` (AUTO/ANY/NONE ≈ auto/required/none), and the same request→model-decides→execute→send-result→synthesize loop. Everything in this project transfers directly to OpenAI's SDK with only parameter-name differences (documented inline in each script).

**A real bug found and fixed during this project:** `format_currency`'s original implementation type-hinted `amount: float` but never validated this at runtime — Python type hints are not enforced automatically. Calling it with a string amount (exactly the kind of malformed argument a model could plausibly generate) raised an uncaught `ValueError` from the f-string formatting itself, crashing the script. This is documented in detail in Part 4 below and fixed with explicit runtime validation, applying Day 9's "never trust input, always validate" principle to function *arguments* rather than LLM-generated JSON output.

---

## Part 1: The Function Calling Specification

**Tools array structure:** a list of `types.Tool` objects, each wrapping one or more `types.FunctionDeclaration`s. Each declaration has a `name`, a `description` (critical — this is what the model reads to decide *when* to call the tool, not just *how*), and a `parameters` schema (identical structure to Day 9's JSON schema work: `type: OBJECT`, `properties`, `required`).

**The `tool_choice` parameter (Gemini: `tool_config.function_calling_config.mode`):** `tool_choice_modes_demo.py` demonstrates all three modes against the identical question:

| Mode | OpenAI equivalent | Behavior |
|---|---|---|
| AUTO | tool_choice="auto" | Model decides per-request whether any tool is needed — the default, used throughout this project's other scripts |
| ANY | tool_choice="required" | Model is forced to call some tool, even on a question where no tool is actually useful |
| NONE | tool_choice="none" | Tools remain visible in the schema but calling is disabled — model must answer directly |

Testing all three against a deliberately tool-optional question ("tell me something interesting about the number 42") makes the difference directly observable: AUTO should answer directly (no tool is genuinely needed), ANY should force a call anyway (potentially an unhelpful one — a real, documented trade-off of this mode), and NONE should always answer directly regardless of tool relevance.

**JSON schema for function definitions:** `tools.py` defines all 4 required tools' schemas explicitly using `types.Schema(type="OBJECT", properties={...}, required=[...])` — the exact same schema mechanism verified in Day 9's structured-output work, now describing function *parameters* instead of a response *shape*.

---

## Part 2: The Four Custom Tools

`tools.py` implements all 4 required tools, each with a pure Python function (fully testable, no API dependency) kept deliberately separate from its schema declaration:

1. **get_current_time(timezone_name)** — returns real current time computed from `datetime.now(timezone.utc)` plus a fixed UTC-offset table (dependency-light, no external timezone library needed).
2. **calculate(expression)** — safely evaluates arithmetic via a character-whitelist + sandboxed `eval()` (the exact same pattern verified in Day 6's ReAct Calculator tool), directly motivated by Day 6's measured finding that LLMs are unreliable at large-number mental arithmetic.
3. **search_database(query)** — mock product lookup using keyword-overlap matching, not exact string matching — the same real fix Day 6's ReAct search tool required after discovering exact-match lookups fail against natural model phrasing.
4. **format_currency(amount, currency_code)** — formats a number per real currency conventions (JPY and PKR correctly use 0 decimal places, not the naive "always 2 decimals" assumption).

**Self-test results (fully executed, no API needed):**
```
get_current_time('Mars/OlympusMons') -> {'error': "Unknown timezone 'Mars/OlympusMons'. ..."}
calculate('123000000 * 34000') -> {'result': 4182000000000}
calculate('import os; os.system("ls")') -> {'error': 'Invalid characters in expression...'}
search_database('USB hub for laptop') -> matched 'usb-c hub' via keyword overlap
format_currency(999.99, 'JPY') -> {'formatted': '¥1,000', ...}   (correctly rounds, 0 decimals)
```

The blocked code-injection attempt (`calculate("import os; os.system('ls')")`) is a genuine security-relevant test, not decorative — a `calculate` tool that passed model-generated strings to a real `eval()` without a character whitelist would be a real code-execution vulnerability.

---

## Part 3: The Full Function Calling Loop

`function_calling_loop.py` implements all 5 required stages, manually (automatic function calling disabled) so every stage is inspectable:

```
1. Send tools + question  ->  client.models.generate_content(..., tools=[ALL_TOOLS])
2. Model picks a function  ->  response.function_calls[0]
3. Execute locally          ->  TOOL_REGISTRY[name][0](**args)
4. Send result back          ->  types.Content(role='tool', parts=[Part.from_function_response(...)])
5. Model synthesizes answer  ->  a second generate_content() call using the extended history
```

Tested against 4 questions spanning all 4 tools individually: current time lookup, a calculation genuinely too large to trust mentally, a database lookup, and a currency formatting request — confirming each tool triggers correctly for its intended question type.

---

## Part 4: Multi-Tool Agent — Chaining 2+ Function Calls

`multi_tool_agent.py` answers: "If I buy 3 wireless keyboards and 2 wireless mice, what's the total cost in PKR?" — a question that genuinely requires 4 chained tool calls, none of which alone is sufficient:

```
1. search_database("wireless keyboard")  -> price
2. search_database("wireless mouse")      -> price
3. calculate("(3 * price1) + (2 * price2)")  -> total in USD
4. format_currency(total, "PKR")           -> final formatted answer
```

This is architecturally the same pattern as Day 6's ReAct Thought->Action->Observation loop, expressed through native function calling instead of a prompted text protocol — the model decides each next call based on what it learned from the previous result, in a genuine loop (up to MAX_TOOL_CALLS=8, the same production safeguard against runaway loops established in Day 6), not a pre-scripted sequence.

---

## Part 5: Edge Case Handling — All 3 Required Scenarios, Verified

`edge_case_handling.py` tests all 3 required edge cases, with 2 of the 3 fully executable without any API call:

**Edge case 2 — invalid argument types (fully tested, no API needed):**

| Test | Result |
|---|---|
| format_currency(amount="a lot", ...) | Real bug found: originally raised uncaught ValueError; fixed to return a clean error dict |
| calculate() with no arguments | TypeError correctly raised and caught: "missing 1 required positional argument" |
| get_current_time(unexpected_arg=...) | TypeError correctly raised and caught: "unexpected keyword argument" |

**Edge case 3 — function returns an error (fully tested, no API needed):** 5 distinct semantic-error cases (unknown timezone, division by zero, incomplete expression, unknown currency code, no database match) — every one returns a clean {"error": ...} dict rather than raising, letting the model explain the failure in natural language rather than crashing the pipeline.

**Edge case 1 — model declines to call a function:** demonstrated via function_calling_loop.py's `if not response.function_calls:` branch, tested with a question no tool is relevant to ("What's the capital of France?"). This is documented as a normal, valid outcome to handle as a first-class code path, not an error condition — a model correctly recognizing it doesn't need a tool is the system working as intended.

**The defense-in-depth principle demonstrated across all 3 cases:** functions never raise uncaught exceptions for semantically-invalid-but-correctly-typed input (they return {"error": ...} instead); genuinely malformed calls (wrong type, missing/extra arguments) are caught one level up in the calling loop and converted to the same {"error": ...} shape — the model never needs to know which category of failure occurred, it just receives a consistent, explainable result either way.

---

## Part 6: Function Calling vs. Regular JSON Mode

Full documentation in `json_mode_vs_function_calling.md`. Core distinction: JSON mode formats an answer the model can already produce from its own knowledge; function calling lets the model ground its answer in real, external, current information. The deciding question: does this require information or computation the model doesn't already have? If no, JSON mode (Day 9); if yes, function calling (Day 10).

The two combine in real production systems: function calling gathers real data, then a JSON-mode-style schema constrains the final synthesized answer — the complete pattern this project's multi_tool_agent.py (function calling half) and Day 9's json_schema_enforcement.py (JSON mode half) together demonstrate.

---

## How Day 10 Connects to Earlier Days

| Earlier concept | Role in Day 10 |
|---|---|
| Day 6: ReAct pattern, tool-matching bug, MAX_TURNS safeguard | Directly reused: keyword-overlap search matching, the turn-limit safeguard, the Thought->Action->Observation shape now expressed as native function calling |
| Day 6: LLMs unreliable at large-number arithmetic | Direct motivation for the calculate tool and its system-instruction warning against mental computation |
| Day 8: Token cost / latency awareness | Directly informs Part 6's "why not use function calling for everything" — every round-trip has a real cost |
| Day 9: JSON schema enforcement, defense-in-depth validation | The exact same schema mechanism now describes function parameters; the same "never trust the API/model promise alone, validate in code" principle now applied to function arguments (Part 5) |
| Day 9: Real, live-discovered model deprecation | Same model (gemini-3.5-flash-lite) and same discipline — discover real issues through testing, document and fix them transparently — applied here to the format_currency type-validation bug |

Day 10 completes the arc from Day 6's prompted, text-protocol ReAct agent to a native, schema-enforced, production-grade function calling agent — the same underlying idea (let the model request real actions, execute them for real, feed results back) now expressed through the API mechanism actually built for it.
