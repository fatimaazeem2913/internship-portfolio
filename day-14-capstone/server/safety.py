"""
safety.py
------------
Requirement #6: "The application shall politely reject abusive or
inappropriate input, prevent harmful or offensive responses, and redirect
unsafe conversations toward safe educational content."

TWO LAYERS OF DEFENSE, matching this project's established pattern (Day
9's dual-layer JSON validation, Day 10's tool argument validation):

  1. A FAST, DETERMINISTIC PRE-FILTER (this module) -- catches blatant
     profanity/abuse via keyword matching BEFORE any API call is made.
     This is cheap, instant, and 100% reliable for what it covers -- it
     also means egregiously abusive input never even reaches (or costs)
     an LLM call.
  2. The ACTIVITY SYSTEM PROMPTS (activities.py) -- instruct the model
     itself to decline and redirect anything subtler the keyword filter
     wouldn't catch (a keyword list can never be exhaustive; natural
     language has too many ways to express something inappropriate).

Neither layer alone is sufficient -- the keyword filter is fast but
crude and easily evaded by rephrasing; the system prompt is more
flexible but not a hard guarantee. Together they're the same
defense-in-depth principle used throughout this internship.
"""

import re

_BLOCKED_PATTERNS = [
    r"\bkill\s+(yourself|you|him|her|them)\b",
    r"\bshut\s+up\b",
    r"\bstupid\s+(bot|ai|robot)\b",
    r"\bi\s+hate\s+you\b",
    r"\bidiot\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

SAFE_REDIRECT_MESSAGE = (
    "Let's keep things kind and fun! I'm not able to respond to that, "
    "but I'd love to help with something else -- want to try a riddle, "
    "a quick quiz question, or ask me something you're curious about?"
)


def is_blatantly_inappropriate(text):
    """
    Fast, deterministic check for the most obvious cases. Returns True
    if the text matches a blocked pattern -- callers should skip the LLM
    call entirely and return SAFE_REDIRECT_MESSAGE directly.
    """
    return any(pattern.search(text) for pattern in _COMPILED_PATTERNS)


if __name__ == "__main__":
    print("=" * 90)
    print("SAFETY PRE-FILTER -- SELF-TEST (pure Python, no API needed)")
    print("=" * 90)

    test_cases = [
        ("What's 2+2?", False),
        ("Can you tell me a riddle?", False),
        ("You are so stupid bot", True),
        ("I hate you", True),
        ("shut up", True),
        ("What's the capital of France?", False),
        ("you're an idiot", True),
        ("Tell me about space", False),
    ]

    passed = 0
    for text, expected_blocked in test_cases:
        result = is_blatantly_inappropriate(text)
        status = "PASS" if result == expected_blocked else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"[{status}] '{text}' -> blocked={result} (expected {expected_blocked})")

    print(f"\n{passed}/{len(test_cases)} test cases passed.")
