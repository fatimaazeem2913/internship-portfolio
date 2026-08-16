"""
answer_checking.py
----------------------
Pure Python fuzzy answer-checking, used by both Brain Buster and Quick
Fire -- deliberately NOT delegated to the LLM (see activities.py's
docstring for the full reasoning).

WHY "FUZZY" AND NOT EXACT MATCH:
A child typing "sun" should be marked correct against the stored answer
"Sun" (case), "sun." (trailing punctuation), or " sun " (stray
whitespace). This is the same principle as Day 6's ReAct search tool
fix: exact-match comparison is too brittle for real input variation.
"""

import re

# Handles the real, likely case of a math question where the stored
# answer is a word ("seven") but a child naturally types the digit ("7"),
# or vice versa -- without this, a genuinely correct answer would be
# marked wrong purely due to numeral-vs-word-form mismatch.
_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}
_WORD_TO_NUMBER = {word: digit for digit, word in _NUMBER_WORDS.items()}


def normalize(text):
    """Lowercase, strip whitespace, remove trailing punctuation, collapse internal whitespace."""
    text = text.strip().lower()
    text = re.sub(r"[.!?,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _number_equivalent_forms(normalized_text):
    """
    Returns a set containing the normalized text plus, if it's a known
    number (digit or word form), its equivalent in the OTHER form --
    so "7" and "seven" are treated as the same answer either direction.
    """
    forms = {normalized_text}
    if normalized_text in _NUMBER_WORDS:
        forms.add(_NUMBER_WORDS[normalized_text])
    if normalized_text in _WORD_TO_NUMBER:
        forms.add(_WORD_TO_NUMBER[normalized_text])
    return forms


def is_correct_answer(user_guess, correct_answer):
    """
    Returns True if user_guess reasonably matches correct_answer.
      1. Exact match after normalization, OR a numeral/word-form match
         (e.g. "7" correctly matches stored answer "seven").
      2. The normalized correct answer (in either numeral or word form)
         appears as a whole word/phrase within the normalized guess
         ("I think it's a sun" -> "sun"; "it's 7" -> "seven").
    """
    norm_guess = normalize(user_guess)
    norm_answer = normalize(correct_answer)

    if not norm_answer:
        return False

    answer_forms = _number_equivalent_forms(norm_answer)
    if norm_guess in answer_forms:
        return True

    for form in answer_forms:
        pattern = r"\b" + re.escape(form) + r"\b"
        if re.search(pattern, norm_guess):
            return True
    return False


if __name__ == "__main__":
    print("=" * 90)
    print("ANSWER CHECKING -- SELF-TEST (pure Python, no API needed)")
    print("=" * 90)

    test_cases = [
        ("sun", "Sun", True),
        ("Sun.", "sun", True),
        ("  sun  ", "sun", True),
        ("I think it's the sun", "sun", True),
        ("sunday", "sun", False),
        ("sunshine", "sun", False),
        ("moon", "sun", False),
        ("seven", "Seven", True),
        ("7", "seven", True),   # numeral vs word form -- now correctly matches
        ("seven", "7", True),   # the reverse direction too
        ("it's 7", "seven", True),
        ("Paris", "paris", True),
        ("", "sun", False),
        ("the answer is elephant", "elephant", True),
    ]

    passed = 0
    for guess, answer, expected in test_cases:
        result = is_correct_answer(guess, answer)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"[{status}] guess='{guess}' vs answer='{answer}' -> correct={result} (expected {expected})")

    print(f"\n{passed}/{len(test_cases)} test cases passed.")
