# Function Calling & Tool Use – Day 10 Internship

## Project Overview

This project was completed as part of Day 10 internship tasks. The objective was to master function calling to build LLMs that interact with external systems — the core mechanism behind AI agents and tool-augmented applications.

Every script uses Gemini (gemini-3.5-flash-lite) as the primary, free API — Gemini's function calling mechanism is structurally equivalent to OpenAI's (a tools array of JSON-schema function declarations, a tool_config mode parameter directly analogous to OpenAI's tool_choice), so every concept transfers directly. A real bug was found and fixed during development: format_currency originally crashed on a wrong-typed argument instead of failing gracefully — documented transparently in REPORT.md Part 5, the same discipline established across Days 6 and 9.

---

## Objectives

- Study the function calling specification: tools array structure, tool_choice parameter, JSON schema for function definitions.
- Define 4 custom tools with proper JSON schemas: get_current_time, calculate, search_database (mock), format_currency.
- Implement the full function calling loop: send tools -> model picks function -> execute locally -> send result back -> model synthesizes response.
- Build a multi-tool agent that chains 2+ function calls to answer a complex question requiring both calculation and data lookup.
- Handle edge cases: model declines to call a function, invalid argument types, function returns an error.
- Document the difference between function calling for structured output vs. regular JSON mode.

---

## Technologies Used

- Python 3
- google-genai (Gemini SDK)

---

## Project Structure

```
day-10-function-calling
|
|-- README.md
|-- REPORT.md
|-- json_mode_vs_function_calling.md
|
|-- tools.py
|-- function_calling_loop.py
|-- multi_tool_agent.py
|-- edge_case_handling.py
|-- tool_choice_modes_demo.py
|
|-- outputs
    |-- function_calling_loop_results.txt   (generated when run locally)
    |-- multi_tool_agent_results.txt          (generated when run locally)
    |-- edge_case_handling_results.txt         (fully generated -- most cases need no API)
    `-- tool_choice_modes_results.txt           (generated when run locally)
```

---

## Tasks Performed

### 1. Function Calling Specification Study

Documented in REPORT.md Part 1: tools array structure, JSON schema for function parameters (identical mechanism to Day 9's structured output schemas), and all three tool_choice/tool_config modes.

### 2. Four Custom Tools

tools.py — get_current_time, calculate, search_database (mock), format_currency, each with a pure Python function fully tested without any API call, kept deliberately separate from its JSON schema declaration.

### 3. Full Function Calling Loop

function_calling_loop.py — all 5 stages implemented manually (automatic function calling disabled) so every stage is visible: send tools, model decides, execute locally, send result back, model synthesizes.

### 4. Multi-Tool Agent

multi_tool_agent.py — chains 4 tool calls (2 database lookups, 1 calculation, 1 currency format) to answer a single complex question, with a MAX_TOOL_CALLS safeguard against runaway loops (Day 6's ReAct lesson).

### 5. Edge Case Handling

edge_case_handling.py — all 3 required edge cases tested, 2 of 3 fully executable without any API call: invalid argument types, function returns an error, model declines to call a function.

### 6. Function Calling vs. JSON Mode

json_mode_vs_function_calling.md — full comparison with a concrete decision test and examples drawn directly from this project.

### 7. tool_choice / tool_config Modes

tool_choice_modes_demo.py — AUTO, ANY, and NONE tested against the identical question, making the behavioral difference directly observable.

---

## Results

- **All 4 tools self-tested and verified**, including 2 deliberate error cases each (unknown timezone, division by zero, unknown currency, no database match) and a blocked code-injection attempt on the calculate tool's eval() sandbox.
- **A real bug found and fixed:** format_currency(amount="a lot", ...) originally raised an uncaught ValueError instead of returning a clean error — fixed with explicit runtime type validation, verified with a rerun showing the graceful {"error": ...} response instead.
- **8 distinct edge-case scenarios tested**, covering both exception-based failures (wrong type, missing/extra arguments) and semantic-error-return failures (invalid input that's still correctly typed) — every one handled without crashing the pipeline.
- **All 5 API-calling scripts confirmed to fail correctly** with a clear "No API key was provided" error when GEMINI_API_KEY is unset.

---

## Observations

- Separating each tool's schema declaration from its actual Python implementation (Part 2) made it possible to fully test and debug the implementations — including finding the real format_currency bug — entirely offline, before spending a single API call.
- The calculate tool's character-whitelist defense against eval() injection is not a decorative security theater — it's a genuine requirement the moment any LLM-generated string reaches a real eval() call, since the model's output is, by definition, untrusted input from the application's perspective.
- The multi-tool agent's 4-call chain (2 lookups -> 1 calculation -> 1 format) is architecturally identical to Day 6's ReAct pattern — the same Thought->Action->Observation shape, just expressed through the API's native mechanism instead of a prompted text protocol the model has to be taught to follow.
- Testing all three tool_choice/tool_config modes against the SAME deliberately tool-optional question was the only way to make the ANY mode's real trade-off (forcing a call even when none is useful) directly visible, rather than just asserting it exists.

---

## Challenges Encountered

- **A real bug found through testing, not anticipated in advance:** format_currency's original amount: float type hint gave a false sense of safety — Python doesn't enforce type hints at runtime, so a string argument reached the f-string formatting code directly and raised an uncaught ValueError, crashing edge_case_handling.py mid-run. Fixed by adding an explicit isinstance() check at the top of the function, converting this into the same graceful {"error": ...} pattern used everywhere else — this is documented in detail rather than silently corrected, since it's a genuinely instructive example of why type hints alone are not validation.
- Designing a question for tool_choice_modes_demo.py that would actually make AUTO/ANY/NONE behave differently required deliberately choosing a tool-optional question rather than an obviously tool-necessary one (like a calculation) — an obviously-necessary question would make AUTO and ANY produce identical behavior, hiding the actual distinction being demonstrated.
- Ensuring search_database used keyword-overlap matching rather than exact-string matching was informed directly by Day 6's ReAct tool-matching bug — building it correctly from the start here, rather than discovering the same failure mode a second time.

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-10-function-calling
```

Install dependencies:
```
pip install google-genai
```

Run the fully offline scripts first (no API key needed):
```
python3 tools.py
python3 edge_case_handling.py
```

Then the live scripts:
```
export GEMINI_API_KEY="your-key-here"
python3 function_calling_loop.py
python3 multi_tool_agent.py
python3 tool_choice_modes_demo.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- The complete function calling specification: tools array structure, JSON schema for function parameters, and the three tool_choice/tool_config modes and their real production trade-offs.
- How to build tools that fail gracefully — returning clean, explainable error dicts rather than raising exceptions — and why this matters specifically because an LLM, not a trusted internal caller, is the one generating the arguments.
- Why Python type hints alone do not provide runtime safety, discovered through a real bug rather than an abstract warning.
- How to build a genuine multi-tool agentic loop with a production safeguard against runaway tool-calling, directly building on Day 6's ReAct pattern.
- A clear, decision-oriented understanding of when function calling is necessary (live data, precise computation, external actions) versus when simpler JSON mode suffices (formatting information the model already has).

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 10
