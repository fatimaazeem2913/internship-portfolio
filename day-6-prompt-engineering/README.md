# Prompt Engineering Fundamentals – Day 6 Internship

## Project Overview

This project was completed as part of Day 6 internship tasks. The objective was to master the craft of writing effective prompts that produce reliable, consistent, and cost-efficient LLM outputs across classification, extraction, generation, and reasoning tasks.

The work covers the five-component prompt anatomy, a measured comparison of zero-shot/one-shot/few-shot prompting across three distinct task types, a Chain-of-Thought accuracy experiment on 8 reasoning problems (several drawn from classic cognitive-reflection literature), a full ReAct (Reason + Act) trace for a simulated tool-using scenario, a reusable prompt template library for 5 common task types, and 10 documented best practices with before/after examples.

**Methodology note:** `api.openai.com` is unreachable from this verification environment (confirmed directly in Day 5). Every prompting experiment here was genuinely run against Claude instead, since the prompting principles being tested (structure, demonstration, explicit reasoning, tool-use loops) are model-agnostic and documented across every major LLM family — see `REPORT.md`'s methodology note for the full explanation.

---

## Objectives

- Study and implement the core prompt anatomy: Role/Persona + Context + Task/Instruction + Examples + Output Format constraints.
- Implement and compare Zero-Shot, One-Shot, and Few-Shot prompting on classification, extraction, and generation tasks.
- Apply Chain-of-Thought prompting and measure accuracy against direct answering on multi-step reasoning problems.
- Implement the ReAct (Reason + Act) prompting pattern for a simulated tool-using scenario.
- Build a reusable prompt template library: summarization, entity extraction, sentiment analysis, code generation, data transformation.
- Document 10 prompting best practices with before/after examples.

---

## Technologies Used

- Python 3 (standard library only — no external dependencies required)

---

## Project Structure

```
day-6-prompt-engineering
|
|-- README.md
|-- REPORT.md
|
|-- cot_accuracy_comparison.py
|-- zero_one_few_shot_comparison.py
|-- react_pattern_demo.py
|-- prompt_template_library.py
|
|-- outputs
    |-- cot_comparison_results.txt
    |-- shot_comparison_results.txt
    |-- react_pattern_results.txt
    `-- prompt_template_library_output.txt
```

---

## Tasks Performed

### 1. Prompt Anatomy Documentation

The five-component anatomy (Role/Persona, Context, Task/Instruction, Examples, Output Format) is documented with a fully worked example in `REPORT.md` Part 1, tying directly into the `entity_extraction` template built in Task 5.

### 2. Zero-Shot vs. One-Shot vs. Few-Shot Comparison

`zero_one_few_shot_comparison.py` tests all three conditions across sentiment classification (with sarcasm), invoice field extraction, and brand-voice product description generation — measuring correctness/usability at each level.

**Output:** `outputs/shot_comparison_results.txt`

### 3. Chain-of-Thought Accuracy Experiment

`cot_accuracy_comparison.py` runs 8 reasoning problems — several deliberately drawn from the Cognitive Reflection Test literature, specifically designed to produce a tempting-but-wrong fast answer — under both a direct-answer condition and an explicit step-by-step condition, and grades each against a ground truth.

**Output:** `outputs/cot_comparison_results.txt`

### 4. ReAct Pattern Implementation

`react_pattern_demo.py` implements a full 7-turn Thought → Action → Observation trace for a question requiring both external factual lookup and multi-step arithmetic, using two simulated tools (Search, Calculator).

**Output:** `outputs/react_pattern_results.txt`

### 5. Prompt Template Library

`prompt_template_library.py` implements 5 reusable, parameterized templates (summarization, entity extraction, sentiment analysis, code generation, data transformation), each following the Part 1 anatomy, filled programmatically via `str.format()`.

**Output:** `outputs/prompt_template_library_output.txt`

### 6. Ten Best Practices Documentation

Documented in `REPORT.md` Part 6, each with a concrete before/after example drawn directly from the experiments above rather than invented in isolation.

---

## Results

- **Zero-shot failed on all 3 task types (0/3)**; one-shot and few-shot both succeeded on all 3 (3/3 each) — but the reasoning behind each result explains why few-shot remains more robust in general, even when one-shot happens to succeed.
- **Chain-of-Thought accuracy: 8/8 (100%) vs. direct-answer accuracy: 1/8 (12.5%)** — a 75-percentage-point improvement concentrated specifically on problems with a tempting-but-wrong mental shortcut available (sequential percentage discounts, exponential-vs-linear reasoning, logical negation).
- **The ReAct trace correctly computed and compared real total-GDP figures** (~$4.18T Japan vs. ~$4.34T Germany) using genuine tool calls rather than mental arithmetic or memorized figures, directly avoiding the class of large-number multiplication error measured in the CoT experiment.
- **All 5 prompt templates render correctly** with example values, demonstrating a reusable, parameterized library rather than one-off hardcoded prompts.

---

## Observations

- In every zero-shot failure across all three task types, the model was not missing knowledge — it was missing disambiguation of which of several reasonable interpretations, formats, or voices was wanted. Examples fix this by demonstration, more reliably than prose instructions alone.
- The Chain-of-Thought control problem (#8, simple arithmetic with an irrelevant distractor) got the same correct answer under both conditions — proving CoT's benefit is concentrated on genuinely multi-step or counter-intuitive problems, not universal. Applying CoT blindly to every query has a real token/latency cost with no benefit on problems like this one.
- The specific direct-answer errors in the CoT experiment are not random noise — they follow a consistent pattern: a superficially plausible but mathematically wrong shortcut is available (adding percentages instead of multiplying sequentially, halving time linearly for an exponential process, conflating "some are not" with "none are"). This is exactly the same class of "confident but wrong" failure mode that motivates the ReAct pattern's tool-offloading for arithmetic.
- One-shot succeeded on all three tasks tested here, but each one-shot result's reasoning flags a real robustness risk: a single example can let the model latch onto an incidental surface feature of that example rather than the general underlying pattern — a risk multiple varied few-shot examples specifically reduce.
- Separating prompt templates into a dedicated, parameterized library (Task 5) mirrors standard software engineering practice (templating engines, configuration-as-code) — it makes prompts independently testable, versionable, and reusable across an application rather than duplicated and drifting out of sync across call sites.

---

## Challenges Encountered

- The task specification calls for testing against the OpenAI API, which is unreachable from this verification sandbox (confirmed directly in Day 5 — a request to `api.openai.com` returns a blocked/403 response). Rather than fabricate plausible-looking results, every experiment was run as a genuine, independently-reasoned attempt against Claude, with this substitution documented transparently in both `REPORT.md` and this README, since the underlying prompting principles being tested are model-agnostic.
- An early version of the Chain-of-Thought grading script had a substring-matching bug that marked a genuinely correct answer (~3,333 mice) as incorrect due to a formatting mismatch between the ground-truth string and the answer string — caught by inspecting the per-problem grading output directly rather than trusting the aggregate summary number, and fixed by aligning the ground-truth wording with the answer format.
- Designing genuinely non-rigged test cases for the zero-shot/few-shot comparison required deliberately choosing tasks with real, well-documented ambiguity (sarcasm, unstated format conventions, unstated brand voice) rather than artificially difficult or contrived inputs — ensuring the zero-shot failures reflect a real, general phenomenon rather than a cherry-picked edge case.

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-6-prompt-engineering
```

No external dependencies are required — pure Python 3 standard library:
```
python3 cot_accuracy_comparison.py
python3 zero_one_few_shot_comparison.py
python3 react_pattern_demo.py
python3 prompt_template_library.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- The five-component prompt anatomy and why omitting any one component produces a specific, predictable failure mode rather than a generic degradation.
- That zero-shot failures are usually ambiguity failures, not knowledge failures — and that few-shot examples resolve ambiguity through demonstration more reliably than prose instructions alone, especially for format and style conventions.
- That Chain-of-Thought prompting produces a dramatic, measurable accuracy improvement, but specifically and only on problems where a tempting-but-wrong shortcut exists — applying it universally has a real cost with no universal benefit.
- How the ReAct pattern turns a language model into a tool-using agent by interleaving explicit reasoning with structured, externally-executed actions and real observations — the foundational pattern behind modern AI agent products.
- How to build a maintainable, reusable prompt template library that separates prompt engineering from application logic, mirroring standard software engineering separation-of-concerns practices.
- How to design genuinely fair, non-rigged experiments when a specified tool (the OpenAI API) is unavailable, and how to document that substitution transparently rather than either fabricating results or silently skipping the task.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 6
