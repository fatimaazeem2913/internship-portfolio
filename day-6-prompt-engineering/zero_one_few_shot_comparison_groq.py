"""
zero_one_few_shot_comparison_groq.py
------------------------------------------
FREE VERSION: uses Groq's free API tier instead of OpenAI. Fully
OpenAI-compatible SDK -- only base_url and model name change.

SETUP (run locally):
    pip install openai
    Get a free key at https://console.groq.com/keys (no credit card needed)
    export GROQ_API_KEY="gsk_...your-key-here..."
    python3 zero_one_few_shot_comparison_groq.py
"""

import os
from openai import OpenAI
from prompt_loader import load_prompts

MODEL = "llama-3.3-70b-versatile"

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

TEST_INPUTS = {
    "review": "Oh great, ANOTHER update that breaks the login page. Exactly what I needed today.",
    "invoice_text": "Invoice ref 88213 -- billed to Marcus Aurelius Consulting, due 3rd of Nov, amount owing: two thousand four hundred and fifty dollars",
    "product_description": "a wireless mechanical keyboard with hot-swappable switches and a 4000mAh battery",
}


def call_openai(prompt, temperature=0.0):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    prompts = load_prompts("prompts/shot_prompts.md")
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out(f"ZERO/ONE/FEW-SHOT COMPARISON -- REAL Groq Free API ({MODEL})")
    out("=" * 100)

    tasks = [
        ("CLASSIFICATION", "review", TEST_INPUTS["review"]),
        ("EXTRACTION", "invoice_text", TEST_INPUTS["invoice_text"]),
        ("GENERATION", "product_description", TEST_INPUTS["product_description"]),
    ]

    for task_name, placeholder, value in tasks:
        out(f"\n\n{'#'*100}")
        out(f"TASK: {task_name}")
        out(f"{'#'*100}")
        out(f"\nInput: {value}")

        for shot_type in ["ZERO_SHOT", "ONE_SHOT", "FEW_SHOT"]:
            key = f"{task_name}_{shot_type}"
            prompt = prompts[key].format(**{placeholder: value})
            response = call_openai(prompt, temperature=0.0)
            out(f"\n--- {shot_type} ---")
            out(f"Response: {response}")

    with open("outputs/shot_comparison_groq_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n\nSaved to outputs/shot_comparison_groq_results.txt")
