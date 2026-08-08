"""
production_prompts_demo.py
------------------------------
Runs all four production prompt types -- structured JSON generation,
unstructured text parsing, code generation, document summarization --
loaded from prompts/production_prompts.md (kept separate from this
application logic, per Day 6's Best Practice #10).

SETUP:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python3 production_prompts_demo.py
"""

import os
from google import genai
from google.genai import types
from prompt_loader import load_prompts
from json_schema_enforcement import safe_parse_json

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

TEST_INPUTS = {
    "product_description": "A stainless steel insulated water bottle, 32oz, keeps drinks cold for 24 hours",
    "email_text": (
        "Hi, this is the third time I'm emailing about my account being charged twice "
        "for the same subscription last week. I need this fixed urgently, I'm a small "
        "business owner and this is affecting my cash flow. -- Priya Sharma"
    ),
    "language": "Python",
    "task_description": "returns the running median of a stream of numbers as each new number arrives",
    "edge_case": "the stream is empty when median() is first called",
    "n_sentences": "2",
    "document": (
        "The James Webb Space Telescope, launched in December 2021, is the largest and "
        "most powerful space telescope ever built. Positioned at the second Lagrange "
        "point roughly 1.5 million kilometers from Earth, it observes primarily in the "
        "infrared spectrum, allowing it to see through cosmic dust clouds and observe "
        "some of the earliest galaxies formed after the Big Bang. The telescope's "
        "primary mirror consists of 18 hexagonal gold-coated beryllium segments spanning "
        "6.5 meters in diameter."
    ),
}


def run_prompt(system_key, user_key, temperature=0.2, **format_kwargs):
    prompts = load_prompts("prompts/production_prompts.md")
    system_prompt = prompts[system_key].format(**format_kwargs)
    user_prompt = prompts[user_key].format(**format_kwargs)

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    return response.text


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("FOUR PRODUCTION PROMPT TYPES")
    out("=" * 90)

    out("\n" + "#" * 90)
    out("1. STRUCTURED JSON GENERATION")
    out("#" * 90)
    result = run_prompt("JSON_GENERATION_SYSTEM", "JSON_GENERATION_USER",
                         product_description=TEST_INPUTS["product_description"])
    out(f"Input: {TEST_INPUTS['product_description']}")
    out(f"Output: {result}")
    parsed_ok, parsed = safe_parse_json(result)
    if parsed_ok:
        out(f"Parses as valid JSON: True -- {parsed}")

    out("\n" + "#" * 90)
    out("2. UNSTRUCTURED TEXT PARSING")
    out("#" * 90)
    result = run_prompt("TEXT_PARSING_SYSTEM", "TEXT_PARSING_USER",
                         email_text=TEST_INPUTS["email_text"])
    out(f"Input email: {TEST_INPUTS['email_text'][:100]}...")
    out(f"Output: {result}")

    out("\n" + "#" * 90)
    out("3. CODE GENERATION")
    out("#" * 90)
    result = run_prompt("CODE_GENERATION_SYSTEM", "CODE_GENERATION_USER",
                         language=TEST_INPUTS["language"],
                         task_description=TEST_INPUTS["task_description"],
                         edge_case=TEST_INPUTS["edge_case"])
    out(f"Task: {TEST_INPUTS['task_description']}")
    out(f"Output:\n{result}")

    out("\n" + "#" * 90)
    out("4. DOCUMENT SUMMARIZATION")
    out("#" * 90)
    result = run_prompt("SUMMARIZATION_SYSTEM", "SUMMARIZATION_USER",
                         n_sentences=TEST_INPUTS["n_sentences"],
                         document=TEST_INPUTS["document"])
    out(f"Document length: {len(TEST_INPUTS['document'])} characters")
    out(f"Summary ({TEST_INPUTS['n_sentences']} sentences requested): {result}")

    with open("outputs/production_prompts_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/production_prompts_results.txt")
