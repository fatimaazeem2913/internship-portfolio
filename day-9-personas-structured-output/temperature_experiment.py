"""
temperature_experiment.py
-----------------------------
Runs the SAME prompt multiple times at different temperatures, on two
different task TYPES -- one where determinism is desirable (a factual/
structured task) and one where variety is desirable (a creative task) --
to make concrete the production guidance: match temperature to task type,
not a single global default.

SETUP:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python3 temperature_experiment.py
"""

import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

FACTUAL_SYSTEM = "You classify the sentiment of a review as exactly one word: Positive, Negative, or Neutral."
FACTUAL_PROMPT = "Review: \"The product arrived on time and works exactly as described.\""

CREATIVE_SYSTEM = "You are a creative copywriter."
CREATIVE_PROMPT = "Write a one-sentence tagline for a new brand of hiking boots."

TEMPERATURES = [0.0, 0.7, 1.5]
RUNS_PER_TEMPERATURE = 3


def run(system, prompt, temperature):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
        ),
    )
    return response.text.strip()


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("TEMPERATURE EXPERIMENT: FACTUAL TASK vs. CREATIVE TASK")
    out("=" * 90)

    out("\n" + "#" * 90)
    out("TASK TYPE 1: FACTUAL CLASSIFICATION (determinism is desirable)")
    out("#" * 90)
    out(f"Prompt: {FACTUAL_PROMPT}\n")

    for temp in TEMPERATURES:
        out(f"--- Temperature = {temp} ---")
        results = [run(FACTUAL_SYSTEM, FACTUAL_PROMPT, temp) for _ in range(RUNS_PER_TEMPERATURE)]
        for i, r in enumerate(results, 1):
            out(f"  Run {i}: {r}")
        unique_results = len(set(results))
        out(f"  Unique outputs across {RUNS_PER_TEMPERATURE} runs: {unique_results}")
        out("")

    out("\n" + "#" * 90)
    out("TASK TYPE 2: CREATIVE COPYWRITING (variety is desirable)")
    out("#" * 90)
    out(f"Prompt: {CREATIVE_PROMPT}\n")

    for temp in TEMPERATURES:
        out(f"--- Temperature = {temp} ---")
        results = [run(CREATIVE_SYSTEM, CREATIVE_PROMPT, temp) for _ in range(RUNS_PER_TEMPERATURE)]
        for i, r in enumerate(results, 1):
            out(f"  Run {i}: {r}")
        unique_results = len(set(results))
        out(f"  Unique outputs across {RUNS_PER_TEMPERATURE} runs: {unique_results}")
        out("")

    out("\n" + "=" * 90)
    out("PRODUCTION GUIDANCE: WHEN TO USE EACH TEMPERATURE")
    out("=" * 90)
    out("""
LOW temperature (0.0-0.2) -- use for:
  - Classification, extraction, structured data generation (Day 9's JSON
    schema task) -- you want the SAME correct answer every time
  - Code generation -- you want the most statistically likely (usually
    most conventional, most correct) code, not a "creative" variant
  - Anything feeding into automated downstream processing, where
    inconsistent output format would break a pipeline
  - RAG/factual Q&A (Day 7's finding) -- there's one correct,
    context-supported answer; don't introduce unnecessary variation

MEDIUM temperature (0.5-0.8) -- use for:
  - General conversational assistants -- natural-sounding variation in
    phrasing without drifting into randomness
  - Summarization where slight rewording between runs is acceptable

HIGH temperature (1.0+) -- use for:
  - Creative writing, brainstorming, marketing copy, taglines -- exactly
    where you WANT different runs to produce genuinely different ideas
  - A/B testing multiple creative variants from a single prompt

THE CORE PRINCIPLE (Day 5/6, confirmed again here): temperature controls
how much the probability distribution over next tokens is sharpened
(low T) or flattened (high T) before sampling. On a FACTUAL task there is
one correct answer, so sharpening toward it is exactly what you want. On
a CREATIVE task there are many EQUALLY VALID answers, so flattening
toward diversity is exactly what you want. The mechanism is identical
either way -- only which behavior is DESIRABLE changes with the task.
""")

    with open("outputs/temperature_experiment_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/temperature_experiment_results.txt")
