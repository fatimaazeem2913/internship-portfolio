# System Personas, Roles & Structured Output – Day 9 Internship

## Project Overview

This project was completed as part of Day 9 internship tasks. The objective was to control model behavior precisely by separating context across system/user/assistant roles and enforcing structured JSON output formats — turning Day 6's prompt-engineering principles into tested, reusable, production-style Python code.

Every script uses Gemini as the primary, free API (Day 8's established pattern) — Google AI Studio issues a key with no credit card required. All six scripts were verified to fail correctly and informatively without credentials, and the JSON schema validation logic — pure Python, no API dependency — was fully executed against 7 real test cases (2 valid, 5 deliberately broken in distinct realistic ways), with every failure mode caught correctly.

---

## Objectives

- Split the messages array into system, user, and assistant roles.
- Construct a strict system prompt mandating JSON-schema-only responses, with deviation handled in code.
- Build a few-shot demonstration array inside the message payload.
- Build production prompts for: structured JSON generation, unstructured text parsing, code generation, document summarization.
- Experiment with temperature settings for deterministic vs. creative outputs; document when to use each.
- Implement a persona-switching system across 3 personas (formal, casual, technical).
- Compare Claude, GPT, and Gemini models on context window, token limits, and pricing.

---

## Technologies Used

- Python 3
- google-genai (Gemini SDK — primary, free API used for all live demos)
- openai (reference-compatible; the same patterns apply for local use with a billed account)

---

## Project Structure

```
day-9-personas-structured-output
|
|-- README.md
|-- REPORT.md
|-- model_comparison.md          (reused, verified in Day 8)
|
|-- prompts
|   |-- production_prompts.md
|   `-- personas.md
|
|-- prompt_loader.py
|-- token_cost_calculator.py     (reused from Day 8)
|
|-- roles_and_messages_demo.py
|-- json_schema_enforcement.py
|-- few_shot_payload_demo.py
|-- production_prompts_demo.py
|-- temperature_experiment.py
|-- persona_switcher.py
|
|-- outputs
    |-- json_validation_test_results.txt      (fully executed, no API needed)
    |-- roles_and_messages_results.txt          (generated when run locally)
    |-- few_shot_payload_results.txt             (generated when run locally)
    |-- production_prompts_results.txt            (generated when run locally)
    |-- temperature_experiment_results.txt         (generated when run locally)
    `-- persona_switcher_results.txt                (generated when run locally)
```

---

## Tasks Performed

### 1. System / User / Assistant Role Separation

roles_and_messages_demo.py builds a real multi-turn conversation with a persistent system instruction and genuine user/assistant history, including a built-in verification test: the model must simultaneously respect a system-level constraint (never invent pricing) and use assistant-role history as real context (continuing the offline-mode conversation coherently).

### 2. Strict JSON Schema Enforcement

json_schema_enforcement.py implements two layers of defense: Gemini's native response_json_schema (API-level, constrains sampling itself) plus independent code-level validation (safe_parse_json + validate_against_schema). Fully tested against 7 cases.

**Output:** outputs/json_validation_test_results.txt

### 3. Few-Shot Demonstration in the Message Payload

few_shot_payload_demo.py builds few-shot examples as genuine alternating conversation turns (not prose in the system prompt), on a date-normalization task deliberately including an ambiguous case (03/09/2026) that only the demonstrated convention resolves correctly.

### 4. Four Production Prompt Types

production_prompts_demo.py — structured JSON generation, unstructured text parsing, code generation (with a stated edge case), and document summarization (with exact sentence-count and fact-preservation requirements).

### 5. Temperature Experiments

temperature_experiment.py runs the same prompt 3 times at 3 temperatures, on both a factual task and a creative task, measuring output diversity directly (counting unique results) rather than describing the effect abstractly.

### 6. Persona-Switching System

persona_switcher.py defines 3 personas and answers an identical question under each, plus a PersonaSession class demonstrating live, runtime persona toggling within a single session.

### 7. Model Comparison

model_comparison.md, carried forward from Day 8 (verified against official pricing pages, current as of August 2026).

---

## Results

- **JSON validation logic: 7/7 test cases handled correctly** — 2 valid cases correctly passed through, 5 deliberately broken cases each caught with a distinct, correct, actionable error message (missing field, invalid enum, out-of-range value, malformed JSON, wrong type).
- **All 6 API-calling scripts confirmed to fail correctly** with a clear "No API key was provided" error when GEMINI_API_KEY is unset — proving the error-handling path works as intended before any script is run against a real key.
- **Both prompt .md files parse correctly** via prompt_loader.py, confirmed by running it directly — 8 sections in production_prompts.md, 4 in personas.md, all found and validated.
- **The markdown-code-fence deviation case** (a model wrapping JSON in fences despite being told not to) was specifically included and correctly handled — this is a real, commonly-observed deviation, not a hypothetical edge case.

---

## Observations

- Building the few-shot examples as actual conversation turns (Part 3), rather than as prose inside a system prompt, is architecturally cleaner — the model processes them via the exact same mechanism as real conversation history (Part 1), rather than treating them as a special instructional case it needs to recognize separately.
- The JSON validation pipeline's design principle — parse first, then validate structure/types/values as fully separate, distinct checks — meant every one of the 5 broken test cases produced a genuinely different, specific error message rather than one generic "invalid input" failure. This distinction matters in production: a specific error message is debuggable; a generic one isn't.
- Testing temperature on two contrasting task types (factual vs. creative) in the same script made the "match temperature to task" principle concrete and measurable (counting unique outputs), rather than asserting it as a rule to take on faith.
- Isolating persona effects by holding the question constant and varying only the system prompt is the same experimental control used throughout this internship (Day 4's isolated-token FFN test, Day 6's task-type-held-constant shot comparisons) — change exactly one variable, observe the effect, attribute it correctly.

---

## Challenges Encountered

- Ensuring the few-shot examples in few_shot_payload_demo.py genuinely tested convention-following (not just easy pattern-matching) required deliberately including an ambiguous test case (03/09/2026) alongside an unambiguous demonstrating example (12/25/2025) — an earlier version without the ambiguous case wouldn't have actually distinguished "the model learned the convention" from "the model got lucky on easy inputs."
- Designing the JSON schema validation test cases required thinking through distinct realistic failure modes (missing field vs. wrong type vs. invalid enum vs. out-of-range value vs. malformed syntax vs. markdown-wrapped output) rather than testing the same failure shape repeatedly — each of the 7 test cases exercises a genuinely different code path in process_llm_response().
- Building the persona test to isolate the persona variable required holding literally everything else constant (same question, same temperature, same model) — an earlier instinct to use different example questions per persona would have made it impossible to attribute output differences to the system prompt alone.

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-9-personas-structured-output
```

Install dependencies:
```
pip install google-genai
export GEMINI_API_KEY="your-key-here"
```

Run the fully offline script first (no API key needed):
```
python3 json_schema_enforcement.py
python3 prompt_loader.py
```

Then the live scripts:
```
python3 roles_and_messages_demo.py
python3 few_shot_payload_demo.py
python3 production_prompts_demo.py
python3 temperature_experiment.py
python3 persona_switcher.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- How to structure a real multi-turn conversation across system/user/assistant roles in code, and how to design a test that actually proves both the system constraint and the conversation history are being used, rather than just asserting they are.
- Why production-grade JSON handling requires two independent layers of defense (API-level schema constraints AND code-level validation), and how to write validation logic that produces distinct, actionable errors for each realistic failure mode rather than one generic catch-all.
- Why embedding few-shot examples as genuine message-history turns is architecturally cleaner than prose-based instruction, and how to design a test case that actually distinguishes convention-learning from lucky pattern-matching.
- How to build reusable, parameterized production prompt templates for four genuinely different task types, all loaded from one shared, version-controlled source.
- How to design an experiment that makes an abstract principle ("match temperature to task") into a measurable, countable result.
- How to build a runtime-toggleable persona system as real, reusable code (a class with a .switch_to() method) rather than a one-off script.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 9
