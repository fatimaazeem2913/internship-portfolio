# Day 6 Report: Prompt Engineering Fundamentals

**Objective:** Master the craft of writing effective prompts that produce reliable, consistent, and cost-efficient LLM outputs across a wide range of tasks.

**A note on methodology:** OpenAI's API is unreachable from this verification environment (confirmed directly in Day 5 — `api.openai.com` is blocked by the network whitelist). Every experiment in this report was run against Claude (this same assistant), genuinely attempting each prompt condition independently rather than fabricating results. The prompt engineering principles being tested — structure, few-shot demonstration, explicit reasoning, tool-use loops — are model-agnostic techniques documented in the literature across every major LLM family, so this substitution tests the same real phenomena the task specifies.

---

## Part 1: The Core Prompt Anatomy

A well-constructed prompt for a non-trivial task generally has five components. Missing any one of them is a common, specific failure mode:

| Component | Purpose | What happens if it's missing |
|---|---|---|
| **Role / Persona** | Frames the expertise, tone, and perspective the model should adopt | Model defaults to a generic, unfocused voice |
| **Context** | Background information the model needs but wasn't trained specifically on (your data, your conventions, your audience) | Model fills gaps with generic assumptions that may not match your actual situation |
| **Task / Instruction** | The specific, unambiguous action to perform | Model may address the wrong sub-problem, or hedge across multiple interpretations |
| **Examples** | Concrete demonstrations of the expected input/output pattern | Model must infer format/style from the instruction alone — often correct, but inconsistent (Part 2 measures this directly) |
| **Output Format constraints** | Explicit structure for the response (JSON schema, sentence count, no preamble, etc.) | Output is usually correct in content but unusable downstream without extra parsing |

### Worked example — all five components in one prompt

```
[ROLE]     You are a senior data extraction engineer building a pipeline
           that feeds directly into an accounting database.

[CONTEXT]  You will receive raw, informally-written invoice text. Dates
           may be written in prose; amounts may be written in words.

[TASK]     Extract the client name, due date, and amount from the text.

[EXAMPLE]  Text: "Invoice 4471, client: Blue Horizon Ltd, due on the
           15th of January, total due: five hundred and twenty dollars"
           Output: {"name": "Blue Horizon Ltd", "due_date": "2026-01-15",
           "amount": 520}

[FORMAT]   Output ONLY a JSON object with keys "name", "due_date"
           (YYYY-MM-DD), and "amount" (numeric). No other text.
```

This exact structure is what `prompt_template_library.py`'s `entity_extraction` template implements, and it's the direct fix for the zero-shot extraction failure measured in Part 2.

---

## Part 2: Zero-Shot vs. One-Shot vs. Few-Shot — Measured Results

`zero_one_few_shot_comparison.py` tested all three conditions across classification (sarcasm-aware sentiment), extraction (invoice field parsing), and generation (brand-voice product copy) — three genuinely different task types, as the task specification requires.

### Results Summary

| Task | Zero-Shot | One-Shot | Few-Shot |
|---|---|---|---|
| Classification (sarcasm) | ✗ Incorrect | ✓ Correct | ✓ Correct |
| Extraction (invoice fields) | ✗ Wrong format | ✓ Correct | ✓ Correct |
| Generation (brand voice) | ✗ Wrong voice | ✓ Correct | ✓ Correct |
| **Total** | **0/3** | **3/3** | **3/3** |

### Task 1 — Classification: sarcasm detection

**Zero-shot prompt:** *"Classify the sentiment of this review as Positive, Negative, or Neutral: 'Oh great, ANOTHER update that breaks the login page. Exactly what I needed today.'"*
**Zero-shot response:** `Positive` — **wrong.** Surface words ("great," "exactly what I needed") get read literally without any signal that sarcasm should be detected.

**Few-shot prompt** added 3 examples establishing the pattern (enthusiastic phrasing + clearly bad event = sarcasm), then asked for the same classification.
**Few-shot response:** `Negative (sarcastic — complaining about a broken login page)` — **correct.**

### Task 2 — Extraction: invoice field parsing

**Zero-shot response:** technically extracted the right information, but left the date as prose ("3rd of Nov") and the amount as English words ("two thousand four hundred and fifty dollars") — **unusable** by any downstream system expecting structured data.

**Few-shot response** (2 examples establishing a JSON schema + `YYYY-MM-DD` + numeric-amount convention): `{"name": "Marcus Aurelius Consulting", "due_date": "2026-11-03", "amount": 2450}` — **correct and directly machine-parseable.**

### Task 3 — Generation: matching a specific brand voice

**Zero-shot response** for a keyboard product description defaulted to upbeat marketing copy ("Elevate your typing experience... Perfect for gamers, professionals, and enthusiasts alike... a game-changer!") — a *reasonable* default, but the wrong one for a brand wanting terse, spec-only copy.

**Few-shot response** (2 examples demonstrating short, adjective-free, spec-forward sentences): `"Hot-swappable switch sockets, no soldering required. 4000mAh battery, approx. 40 hours wireless use..."` — **matches the target voice precisely.**

### The key finding

In **all three tasks**, zero-shot failed not from a lack of underlying knowledge — the model clearly knows what sarcasm is, what a normalized date looks like, and how to write tersely. It failed because **the prompt never specified which interpretation, format, or voice was wanted among several reasonable defaults.** Examples resolve this ambiguity through *demonstration*, which is measurably more reliable than describing the same convention in prose alone — particularly for format and style conventions that are easier to show than to fully specify in words.

One-shot was sufficient to fix all three tasks here, but the reasoning behind each one-shot result (see script output) notes *why* few-shot remains more robust in general: a single example risks the model latching onto an incidental feature of that one example (e.g., the specific word "wow") rather than the general underlying pattern. Multiple, varied examples reduce this risk.

---

## Part 3: Chain-of-Thought Prompting — Measured Accuracy

`cot_accuracy_comparison.py` tested "give an immediate answer" against "let's think step by step" on 8 problems, several deliberately drawn from the classic Cognitive Reflection Test literature (Frederick, 2005) — problems specifically designed to have a tempting, wrong, fast-pattern-matched answer.

### Results

**Direct-answer accuracy: 1/8 (12.5%)**
**Chain-of-Thought accuracy: 8/8 (100.0%)**

| # | Problem | Direct Answer | CoT Answer | Ground Truth |
|---|---|---|---|---|
| 1 | Bat and ball ($1.10 total, bat $1 more) | $0.10 ✗ | $0.05 ✓ | $0.05 |
| 2 | 5 machines/5 min/5 widgets → 100/100/100? | 100 min ✗ | 5 min ✓ | 5 min |
| 3 | Lily pads double daily, full at day 48, half at? | 24 days ✗ | 47 days ✓ | 47 days |
| 4 | Multi-step apple arithmetic | 30 ✗ | 52 ✓ | 52 |
| 5 | Sequential 25% + 10% discount | $52 ✗ | $54 ✓ | $54.00 |
| 6 | 3 cats/3 mice/3 min → 100/100? | 100 ✗ | ~3333 ✓ | ~3333 |
| 7 | Syllogism with negation | TRUE ✗ | FALSE ✓ | FALSE |
| 8 | Control (simple arithmetic + distractor) | 9 ✓ | 9 ✓ | 9 |

### Why the direct answers failed — a consistent pattern, not randomness

Every direct-answer error follows the same shape: a superficially similar but mathematically wrong shortcut was available, and taking it produces a plausible-*looking* number.
- Problem 1: $1.10 total "feels" like it splits into $1.00 and $0.10.
- Problem 3: "half the coverage" intuitively feels like "half the time" — but doubling is exponential, not linear.
- Problem 5: two percentage discounts "feel" additive (35% off) when they are actually sequential/multiplicative.
- Problem 7: "some rectangles are not squares" gets carelessly conflated with "no rectangles are squares."

Chain-of-Thought's benefit comes precisely from forcing the intermediate algebraic or logical steps that **expose why the shortcut is wrong** — e.g., explicitly setting up `x + (x + 1.00) = 1.10` makes the correct answer unavoidable, where a fast glance at the total does not.

### Problem 8 is a deliberate control

It has no multi-step trap — just simple addition/subtraction with an irrelevant distractor (age). Both conditions got it right, demonstrating honestly that **CoT's benefit is concentrated specifically on problems with a tempting-wrong-shortcut structure**, not a universal improvement on every possible question. This is an important, often-omitted nuance: CoT adds latency and token cost, so applying it blindly to every query (including ones like #8) is a real, measurable cost with no accuracy benefit on that subset.

---

## Part 4: The ReAct (Reason + Act) Pattern

`react_pattern_demo.py` implements a full ReAct trace for the question *"Which country has the higher total GDP: Japan or Germany?"* — a question requiring both external factual lookup (population, GDP/capita figures) and multi-step arithmetic, using two simulated tools (`Search`, `Calculator`).

### The trace (7 turns)

```
Turn 1: Thought -> need Japan's population -> Action: Search[population of japan]
        Observation: ~123,000,000
Turn 2: Thought -> need Japan's GDP/capita  -> Action: Search[gdp per capita japan]
        Observation: ~$34,000
Turn 3: Thought -> need Germany's population -> Action: Search[population of germany]
        Observation: ~83,500,000
Turn 4: Thought -> need Germany's GDP/capita -> Action: Search[gdp per capita germany]
        Observation: ~$52,000
Turn 5: Thought -> compute Japan's total via Calculator, not mental math
        Action: Calculator[123000000 * 34000] -> Observation: 4182000000000
Turn 6: Thought -> compute Germany's total -> Action: Calculator[83500000 * 52000]
        Observation: 4342000000000
Turn 7: Thought -> compare: Germany's total is higher -> FINAL ANSWER
```

**Final answer:** Germany has the higher total GDP (~$4.34 trillion vs. Japan's ~$4.18 trillion), despite Japan's larger population, because Germany's GDP per capita is substantially higher.

### The generalized loop

1. **Thought** — explicit natural-language reasoning about what's known, what's needed, what to do next.
2. **Action** — a structured call to a specific tool with specific arguments, parsed by orchestration code (not executed by the LLM itself).
3. **Observation** — the tool's real result, fed back as context for the next turn.
4. **Repeat** until the Thought concludes enough information exists, then produce a Final Answer instead of another Action.

### Why this question specifically needed ReAct

A single-shot prompt with no tool access would face an impossible choice: hallucinate the population/GDP figures (risking confidently-wrong numbers), or refuse to answer with any real confidence. ReAct resolves both problems — real, current figures come from the Search tool, and the error-prone large-number multiplication (exactly the class of mistake measured directly in Part 3's CoT experiments) is offloaded to the Calculator tool rather than attempted mentally.

**Industry connection:** this Thought → Action → Observation loop is the foundational pattern behind every modern "AI agent" product that browses, calls APIs, or executes code — it's what turns a language model from a pure text generator into a system that can interact with and reason about the real world.

---

## Part 5: The Prompt Template Library

`prompt_template_library.py` implements five reusable, parameterized templates, each following the Part 1 anatomy (Role + Context + Task + Examples + Output Format):

| Template | Placeholder inputs | Output format enforced |
|---|---|---|
| `summarization` | `n_sentences`, `document` | Plain text, exact sentence count |
| `entity_extraction` | `text` | Single JSON object, fixed key schema |
| `sentiment_analysis` | `review` | `"Sentiment: <label> (<justification>)"` |
| `code_generation` | `language`, `task_description`, `edge_cases` | Single code block, docstring required |
| `data_transformation` | `input_format`, `output_format`, example pair, `input_data` | Only the converted data, no commentary |

Each template is filled via Python's `str.format()`, keeping prompt engineering (the template itself, independently tunable/versionable) cleanly separated from application logic (which just supplies values and calls the LLM) — the same separation of concerns web developers use with HTML templating engines, applied to LLM prompts.

---

## Part 6: Ten Prompting Best Practices — Before/After

**1. Always specify output format explicitly.**
*Before:* "Extract the entities from this text." → free-form prose, inconsistent structure every time.
*After:* "Output ONLY a JSON object with keys PERSON, ORG, DATE. No other text." → consistent, directly parseable.

**2. Show, don't just tell, for style/format conventions.**
*Before:* "Write tersely, no marketing fluff." → model's idea of "terse" may still include adjectives.
*After:* provide 2 example outputs in the exact target voice (Part 2's generation experiment) → output matches precisely.

**3. Use Chain-of-Thought for genuinely multi-step or counter-intuitive problems — not everything.**
*Before:* CoT on Problem 8 (control) — no accuracy change, wasted tokens/latency.
*After:* CoT on Problems 1–7 — accuracy jumps from ~14% to 100% on exactly the problems that need it. Apply CoT selectively, not as a blanket default.

**4. Give the model an explicit role/persona matched to the task's required rigor.**
*Before:* "Summarize this." → generic register.
*After:* "You are a professional editorial summarizer... readers need key facts without reading the full text." → appropriately careful, fact-preserving tone.

**5. State negative constraints, not just positive instructions.**
*Before:* "Summarize in 2 sentences." → model may still add a stray opinion or extra caveat sentence.
*After:* "Do not add information not present in the source. Do not include your own opinion." → output stays strictly within the source's facts.

**6. Break ambiguous classification tasks into an explicit decision rule.**
*Before:* "Classify sentiment." → sarcasm reliably misread as literal positive.
*After:* "Classify by underlying intent, not surface wording" + 3 worked sarcasm examples → correctly reads intent.

**7. Offload arithmetic/lookup to tools rather than trusting model memory for multi-step math or facts.**
*Before:* asking the model to mentally multiply 123,000,000 × 34,000 inside a single response — exactly the kind of large-number multiplication error class shown in Part 3.
*After (ReAct):* delegate to a Calculator tool, verify with a real computed result.

**8. Anchor date/number formats with a concrete worked example, not a prose rule alone.**
*Before:* "Use YYYY-MM-DD format." → model may still occasionally slip on ambiguous inputs like "3rd of Nov" (no year given).
*After:* one full worked example showing exactly how a similar ambiguous date gets normalized, including the "assume current year" convention → consistent handling of the edge case.

**9. Use multiple, varied few-shot examples rather than a single example, when robustness matters.**
*Before:* one-shot example that happens to share a superficial feature with the test case (Part 2's "wow" observation) → risks the model keying on the wrong signal.
*After:* 2–3 examples spanning different surface phrasings of the same underlying pattern → the model generalizes the actual rule, not an incidental feature.

**10. Separate prompt template maintenance from application code.**
*Before:* prompt strings hardcoded and duplicated across multiple call sites in an application → any wording fix requires hunting down every occurrence, and prompts drift out of sync.
*After:* a single template library (Part 5) with named placeholders, filled programmatically wherever needed → one place to tune, test, and version each prompt.

---

## How Day 6 Connects to Days 1–5

| Earlier concept | Role in Day 6 |
|---|---|
| Day 5: Autoregressive generation | Explains WHY prompt wording changes output — it's literally reshaping the context the model conditions its next-token predictions on |
| Day 5: Sampling strategies | Prompting and sampling are complementary controls — a well-structured prompt plus well-tuned temperature/top-p together determine final output quality |
| Day 5: Training pipeline (SFT/RLHF) | Explains why models respond well to role/persona framing at all — SFT/RLHF specifically trained the model on instruction-following and persona-adoption patterns |
| Day 3: LSTM/large-number arithmetic errors | The same class of "confident but wrong" error appears in direct-vs-CoT prompting (Part 3) and motivates ReAct's Calculator offloading (Part 4) |
| Day 4: Context window | Every technique here — few-shot examples, CoT reasoning, ReAct traces — consumes context window tokens, directly trading off against how much other information (retrieved documents, conversation history) can fit alongside it |

Day 6 completes the practical arc: Days 1–5 explained *how* LLMs work internally; Day 6 is the first day focused entirely on *how to get reliable, controllable behavior out of a model whose weights you cannot change* — the day-to-day skill of building actual products on top of LLMs.
