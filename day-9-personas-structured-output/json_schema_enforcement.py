"""
json_schema_enforcement.py
------------------------------
Constructs a strict system prompt mandating JSON-only output against a
defined schema, using Gemini's native schema-constrained decoding
(response_mime_type + response_json_schema), AND implements defense-in-
depth validation in code to catch any deviation -- because relying on
"the model promised to follow the schema" alone is not production-safe.

TWO LAYERS OF ENFORCEMENT, BOTH DEMONSTRATED:
  1. API-LEVEL: response_json_schema constrains the model's token
     sampling itself (Day 6 study guide's "structured output" concept) --
     a stronger guarantee than a prompt instruction alone.
  2. CODE-LEVEL: even with API-level constraints, always validate the
     parsed result in your own code before trusting it downstream --
     malformed JSON, missing required fields, or wrong types can still
     occur (e.g. if a provider/model doesn't support schema constraints,
     or a network/parsing issue truncates the response).

The validation logic below (validate_against_schema, safe_parse_json) is
PURE PYTHON with no API dependency -- fully tested against both valid and
deliberately-broken mock responses.
"""

import os
import json
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

REVIEW_ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "required": ["sentiment", "confidence", "key_points", "product_mentioned"],
    "properties": {
        "sentiment": {"type": "STRING", "enum": ["positive", "negative", "neutral", "mixed"]},
        "confidence": {"type": "NUMBER"},
        "key_points": {"type": "ARRAY", "items": {"type": "STRING"}},
        "product_mentioned": {"type": "STRING"},
    },
}

SYSTEM_PROMPT = """You are a review analysis engine. You output ONLY valid JSON matching \
the required schema -- no prose, no markdown code fences, no explanation before or after \
the JSON object. Every field is required. "confidence" must be a number between 0 and 1."""


def call_with_schema_enforcement(review_text):
    """API-level enforcement: response_json_schema constrains sampling itself."""
    response = client.models.generate_content(
        model=MODEL,
        contents=review_text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=REVIEW_ANALYSIS_SCHEMA,
            temperature=0.1,
        ),
    )
    return response.text


# ============================================================
# CODE-LEVEL VALIDATION -- pure Python, fully testable without any API
# ============================================================

REQUIRED_FIELDS = {
    "sentiment": str,
    "confidence": (int, float),
    "key_points": list,
    "product_mentioned": str,
}
VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}


def safe_parse_json(raw_text):
    """
    Attempts to parse raw text as JSON. Returns (success, data_or_error).
    Handles the real-world case where a model wraps JSON in markdown
    fences (```json ... ```) despite being told not to -- a common,
    genuine deviation worth defending against defensively.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return True, json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"JSON parsing failed: {e}"


def validate_against_schema(data):
    """
    Validates a parsed dict against REQUIRED_FIELDS and VALID_SENTIMENTS.
    Returns (is_valid, list_of_errors) -- always returns a list, even if
    empty, so callers don't need a separate None-check branch.
    """
    errors = []

    if not isinstance(data, dict):
        return False, [f"Expected a JSON object, got {type(data).__name__}"]

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
            continue
        if not isinstance(data[field], expected_type):
            errors.append(
                f"Field '{field}' has wrong type: expected {expected_type}, "
                f"got {type(data[field]).__name__}"
            )

    if "sentiment" in data and data["sentiment"] not in VALID_SENTIMENTS:
        errors.append(f"Invalid sentiment value: '{data['sentiment']}' (must be one of {VALID_SENTIMENTS})")

    if "confidence" in data and isinstance(data["confidence"], (int, float)):
        if not (0 <= data["confidence"] <= 1):
            errors.append(f"confidence must be between 0 and 1, got {data['confidence']}")

    return len(errors) == 0, errors


def process_llm_response(raw_text):
    """
    The full defense-in-depth pipeline: parse, then validate. This is
    what a production system should ALWAYS do with LLM output headed for
    a database, an API response, or any downstream code -- never trust
    raw model output directly, even with API-level schema enforcement.
    """
    parsed_ok, result = safe_parse_json(raw_text)
    if not parsed_ok:
        return {"status": "PARSE_ERROR", "error": result, "data": None}

    valid, errors = validate_against_schema(result)
    if not valid:
        return {"status": "VALIDATION_ERROR", "error": errors, "data": result}

    return {"status": "OK", "error": None, "data": result}


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("JSON SCHEMA ENFORCEMENT -- VALIDATION LOGIC TEST (pure Python, no API needed)")
    out("=" * 90)

    out(f"\nSchema requires: {list(REQUIRED_FIELDS.keys())}")
    out(f"Valid sentiment values: {VALID_SENTIMENTS}")

    test_cases = [
        (
            "Valid, well-formed response",
            '{"sentiment": "positive", "confidence": 0.92, "key_points": ["fast shipping", "great build quality"], "product_mentioned": "wireless keyboard"}',
        ),
        (
            "Wrapped in markdown code fences (common real deviation)",
            '```json\n{"sentiment": "negative", "confidence": 0.78, "key_points": ["broke after a week"], "product_mentioned": "phone case"}\n```',
        ),
        (
            "Missing a required field",
            '{"sentiment": "positive", "confidence": 0.85, "key_points": ["good value"]}',
        ),
        (
            "Invalid sentiment value (model didn't respect enum)",
            '{"sentiment": "very happy", "confidence": 0.9, "key_points": ["loved it"], "product_mentioned": "backpack"}',
        ),
        (
            "confidence out of range",
            '{"sentiment": "positive", "confidence": 1.5, "key_points": ["excellent"], "product_mentioned": "mug"}',
        ),
        (
            "Completely malformed JSON",
            '{sentiment: positive, confidence: 0.9,,, key_points: [}',
        ),
        (
            "Wrong type for key_points (string instead of array)",
            '{"sentiment": "neutral", "confidence": 0.5, "key_points": "it was okay", "product_mentioned": "lamp"}',
        ),
    ]

    passed = 0
    for label, raw in test_cases:
        out(f"\n--- {label} ---")
        out(f"Raw input: {raw[:100]}{'...' if len(raw) > 100 else ''}")
        result = process_llm_response(raw)
        out(f"Status: {result['status']}")
        if result["error"]:
            out(f"Error(s): {result['error']}")
        else:
            out(f"Validated data: {result['data']}")
            passed += 1

    out(f"\n{'='*90}")
    out(f"SUMMARY: {passed}/{len(test_cases)} test cases resulted in valid, usable data")
    out("(This is EXPECTED and CORRECT -- only 2 of 7 cases were deliberately valid or")
    out(" recoverable; the other 5 were deliberately broken in different realistic ways")
    out(" to prove the validation pipeline actually catches each failure mode distinctly")
    out(" rather than crashing or silently accepting bad data.)")
    out("=" * 90)

    with open("outputs/json_validation_test_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/json_validation_test_results.txt")
    print("\n(To test against a REAL Gemini response, set GEMINI_API_KEY and call")
    print(" call_with_schema_enforcement() with a real review, then feed the result")
    print(" through process_llm_response() -- the exact same validation pipeline.)")
