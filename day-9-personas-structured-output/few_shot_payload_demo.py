"""
few_shot_payload_demo.py
----------------------------
Builds a few-shot demonstration array INSIDE the message payload itself
(as alternating user/assistant turns showing concrete input->output
examples), rather than describing the desired pattern in prose within a
single system instruction.

WHY EXAMPLES-AS-MESSAGES INSTEAD OF EXAMPLES-IN-THE-SYSTEM-PROMPT:
Both work, but there's a real, documented reason production systems often
prefer this "fake conversation history" approach: putting each example as
a genuine user/assistant turn pair means the model processes them using
the EXACT SAME mechanism it uses for real conversation history (Day 9's
roles_and_messages_demo.py) -- it's not a special case the model has to
recognize as "instructional text embedded in a system prompt," it's
literally indistinguishable, at the architecture level, from "this
conversation already happened." This is a more direct application of
Day 6's few-shot principle (demonstration beats description) than
embedding the same examples as prose inside a system instruction.

TASK: date-format normalization -- converting messy, inconsistent date
strings into strict ISO 8601 (YYYY-MM-DD) format. Chosen because it is a
narrow, verifiable, deterministic task -- easy to grade objectively
against a known correct answer, exactly like Day 6's extraction task.
"""

import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "You convert dates into strict ISO 8601 format (YYYY-MM-DD). "
    "Output ONLY the converted date, nothing else. If a year is not "
    "given, assume 2026."
)

FEW_SHOT_EXAMPLES = [
    types.Content(role="user", parts=[types.Part(text="3rd of November")]),
    types.Content(role="model", parts=[types.Part(text="2026-11-03")]),
    types.Content(role="user", parts=[types.Part(text="July 4, 1998")]),
    types.Content(role="model", parts=[types.Part(text="1998-07-04")]),
    types.Content(role="user", parts=[types.Part(text="next Monday (today is 2026-08-03)")]),
    types.Content(role="model", parts=[types.Part(text="2026-08-10")]),
    types.Content(role="user", parts=[types.Part(text="12/25/2025")]),
    types.Content(role="model", parts=[types.Part(text="2025-12-25")]),
]


def run_with_few_shot(test_input):
    """
    Sends the few-shot examples AS message history, followed by the real
    test input as the final user turn -- the model sees this as one
    continuous conversation, not as "examples" vs. "the real question."
    """
    payload = FEW_SHOT_EXAMPLES + [types.Content(role="user", parts=[types.Part(text=test_input)])]
    response = client.models.generate_content(
        model=MODEL,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
        ),
    )
    return response.text.strip()


def run_zero_shot(test_input):
    """Same system instruction, but NO few-shot examples -- for direct comparison."""
    response = client.models.generate_content(
        model=MODEL,
        contents=test_input,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
        ),
    )
    return response.text.strip()


TEST_CASES = [
    ("Feb 14", "2026-02-14"),
    ("the 1st of Jan, 2024", "2024-01-01"),
    ("03/09/2026", "2026-03-09"),
    ("this Friday (today is 2026-08-03)", "2026-08-07"),
]


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("FEW-SHOT EXAMPLES EMBEDDED IN MESSAGE PAYLOAD -- DATE NORMALIZATION")
    out("=" * 90)

    out(f"\nSystem instruction: {SYSTEM_INSTRUCTION}")
    out(f"\nFew-shot payload ({len(FEW_SHOT_EXAMPLES)} turns, {len(FEW_SHOT_EXAMPLES)//2} examples):")
    for turn in FEW_SHOT_EXAMPLES:
        role_label = "USER" if turn.role == "user" else "MODEL"
        out(f"  {role_label}: {turn.parts[0].text}")

    out("\n" + "-" * 90)
    out("TEST CASES: zero-shot (no examples) vs. few-shot (examples as message history)")
    out("-" * 90)

    for test_input, expected in TEST_CASES:
        out(f"\nInput: \"{test_input}\"   (expected: {expected})")
        zero_shot_result = run_zero_shot(test_input)
        few_shot_result = run_with_few_shot(test_input)
        out(f"  Zero-shot result: {zero_shot_result}   {'CORRECT' if zero_shot_result == expected else 'DIFFERS FROM EXPECTED'}")
        out(f"  Few-shot result:  {few_shot_result}   {'CORRECT' if few_shot_result == expected else 'DIFFERS FROM EXPECTED'}")

    out("\n" + "=" * 90)
    out("WHAT TO LOOK FOR IN REAL RESULTS")
    out("=" * 90)
    out("""
The most interesting test case is "03/09/2026" -- this is genuinely
ambiguous (March 9th in MM/DD/YYYY, or September 3rd in DD/MM/YYYY).
Without examples, the model has to guess a convention. WITH the few-shot
examples above -- which included "12/25/2025" -> "2025-12-25" (unambiguous,
since there's no 25th month, proving MM/DD/YYYY) -- the model has a
concrete, demonstrated convention to follow rather than an assumption.
This is the same principle verified in Day 6: examples resolve ambiguity
through demonstration more reliably than a prose rule alone.
""")

    with open("outputs/few_shot_payload_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/few_shot_payload_results.txt")
