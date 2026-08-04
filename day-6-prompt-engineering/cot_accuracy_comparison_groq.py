"""
cot_accuracy_comparison_groq.py
------------------------------------
FREE VERSION: uses Groq's free API tier (no credit card required, ~14,400
requests/day) instead of OpenAI. Groq's API is fully OpenAI-compatible --
same Python SDK, same request/response shape -- only the base_url and
model name change. Everything else about this experiment (the prompts,
the grading logic) is identical to the OpenAI version.

SETUP (run this locally):
    pip install openai
    Get a free key at https://console.groq.com/keys (no credit card needed)
    export GROQ_API_KEY="gsk_...your-key-here..."
    python3 cot_accuracy_comparison_groq.py
"""

import os
from openai import OpenAI
from prompt_loader import load_prompts

MODEL = "llama-3.3-70b-versatile"  # free tier, strong reasoning quality

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

PROBLEMS = [
    {"id": 1, "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "ground_truth": "$0.05"},
    {"id": 2, "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "ground_truth": "5 minutes"},
    {"id": 3, "question": "A patch of lily pads on a lake doubles in size every day. If it takes 48 days for the patch to cover the entire lake, how many days would it take to cover HALF the lake?", "ground_truth": "47 days"},
    {"id": 4, "question": "Sarah has 3 boxes of apples with 12 apples each. She gives away 8 apples, then buys 2 more boxes of 12 apples each. How many apples does she have now?", "ground_truth": "52"},
    {"id": 5, "question": "A shirt costs $80. It is on sale for 25% off. At checkout there is an ADDITIONAL 10% off the already-discounted price. What is the final price?", "ground_truth": "$54.00"},
    {"id": 6, "question": "If 3 cats can catch 3 mice in 3 minutes, how many mice can 100 cats catch in 100 minutes?", "ground_truth": "approximately 3333"},
    {"id": 7, "question": "All squares are rectangles. Some rectangles are not squares. Is the statement 'no rectangles are squares' TRUE or FALSE?", "ground_truth": "FALSE"},
    {"id": 8, "question": "Tom has 5 red marbles and 7 blue marbles. He is 10 years old. He gives 3 red marbles to his sister. How many marbles does Tom have left?", "ground_truth": "9"},
]


def call_openai(prompt, temperature=0.0):
    """Single real Chat Completions call."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def grade(answer, ground_truth):
    def normalize(s):
        return s.lower().replace("$", "").replace(",", "").strip()
    return normalize(ground_truth) in normalize(answer)


if __name__ == "__main__":
    prompts = load_prompts("prompts/cot_prompts.md")
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out(f"CHAIN-OF-THOUGHT vs DIRECT ANSWER -- REAL Groq Free API ({MODEL})")
    out("=" * 100)

    direct_correct = 0
    cot_correct = 0

    for p in PROBLEMS:
        direct_prompt = prompts["DIRECT_PROMPT"].format(question=p["question"])
        cot_prompt = prompts["COT_PROMPT"].format(question=p["question"])

        direct_answer = call_openai(direct_prompt, temperature=0.0)
        cot_answer = call_openai(cot_prompt, temperature=0.0)

        direct_ok = grade(direct_answer, p["ground_truth"])
        cot_ok = grade(cot_answer, p["ground_truth"])
        direct_correct += direct_ok
        cot_correct += cot_ok

        out(f"\n--- PROBLEM {p['id']} ---")
        out(f"Question: {p['question']}")
        out(f"Ground truth: {p['ground_truth']}")
        out(f"\n[DIRECT] {direct_answer}")
        out(f"  {'CORRECT' if direct_ok else 'INCORRECT'}")
        out(f"\n[CoT] {cot_answer}")
        out(f"  {'CORRECT' if cot_ok else 'INCORRECT'}")

    n = len(PROBLEMS)
    out(f"\n\n{'='*100}")
    out("SUMMARY")
    out("=" * 100)
    out(f"Direct-answer accuracy: {direct_correct}/{n} ({100*direct_correct/n:.1f}%)")
    out(f"Chain-of-Thought accuracy: {cot_correct}/{n} ({100*cot_correct/n:.1f}%)")

    with open("outputs/cot_comparison_groq_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved to outputs/cot_comparison_groq_results.txt")
