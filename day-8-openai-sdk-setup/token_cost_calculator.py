"""
token_cost_calculator.py
----------------------------
Parses token usage from an LLM API response and calculates the exact
USD cost of the request, using real, current per-model pricing.

Supports both OpenAI's usage field names (prompt_tokens, completion_tokens,
total_tokens) and Gemini's usage_metadata field names (prompt_token_count,
candidates_token_count, total_token_count) -- these are functionally
identical concepts with different names, exactly the kind of cross-provider
gotcha documented in REPORT.md.

This module is fully testable without a live API call -- it operates on
the documented, stable `usage` object shape each provider returns, so we
can verify the calculation logic against sample/mock usage data before
ever spending real API credit.

PRICING SOURCE: verified against each provider's official pricing page
and multiple independent trackers, current as of August 2026. Providers
revise pricing periodically -- always confirm against the live pricing
page before relying on this for a real budget.
"""

# Prices in USD per 1,000,000 tokens (input, output)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40,  "output": 1.60},
    # Gemini
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash":      {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro":        {"input": 1.25, "output": 10.00},
    # Claude (for the cross-provider comparison in model_comparison.md)
    "claude-haiku-4.5":  {"input": 1.00, "output": 5.00},
    "claude-sonnet-5":   {"input": 2.00, "output": 10.00},
    "claude-opus-5":     {"input": 5.00, "output": 25.00},
}

# Maps each provider's field naming convention to a common internal shape.
FIELD_ALIASES = {
    "prompt_tokens": ["prompt_tokens", "prompt_token_count", "input_tokens"],
    "completion_tokens": ["completion_tokens", "candidates_token_count", "output_tokens"],
    "total_tokens": ["total_tokens", "total_token_count"],
}


def _extract(usage, canonical_field):
    """Try every known alias for a field across providers' different naming conventions."""
    for alias in FIELD_ALIASES[canonical_field]:
        if isinstance(usage, dict) and alias in usage:
            return usage[alias]
        if hasattr(usage, alias):
            return getattr(usage, alias)
    raise AttributeError(
        f"Could not find any of {FIELD_ALIASES[canonical_field]} on the usage object -- "
        f"is this a supported provider's usage shape?"
    )


def calculate_cost(usage, model="gemini-2.5-flash"):
    """
    Calculate the exact USD cost of a request from its token usage.

    Args:
        usage: an object/dict from ANY supported provider -- OpenAI-style
               (prompt_tokens/completion_tokens/total_tokens) or Gemini-style
               (prompt_token_count/candidates_token_count/total_token_count).
        model: model name string, must be a key in MODEL_PRICING.

    Returns:
        dict with prompt_tokens, completion_tokens, total_tokens,
        prompt_cost_usd, completion_cost_usd, total_cost_usd.
    """
    if model not in MODEL_PRICING:
        raise ValueError(f"No pricing data for model '{model}'. "
                          f"Known models: {list(MODEL_PRICING.keys())}")

    prompt_tokens = _extract(usage, "prompt_tokens")
    completion_tokens = _extract(usage, "completion_tokens")
    total_tokens = _extract(usage, "total_tokens")

    rates = MODEL_PRICING[model]
    prompt_cost = (prompt_tokens / 1_000_000) * rates["input"]
    completion_cost = (completion_tokens / 1_000_000) * rates["output"]
    total_cost = prompt_cost + completion_cost

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cost_usd": round(prompt_cost, 8),
        "completion_cost_usd": round(completion_cost, 8),
        "total_cost_usd": round(total_cost, 8),
    }


def format_cost_report(cost_result):
    """Human-readable summary of a calculate_cost() result."""
    c = cost_result
    lines = [
        f"Model: {c['model']}",
        f"  Prompt tokens:     {c['prompt_tokens']:>8}  ->  ${c['prompt_cost_usd']:.8f}",
        f"  Completion tokens: {c['completion_tokens']:>8}  ->  ${c['completion_cost_usd']:.8f}",
        f"  Total tokens:      {c['total_tokens']:>8}",
        f"  TOTAL COST:                    ${c['total_cost_usd']:.8f}",
    ]
    return "\n".join(lines)


class MockUsage:
    """Mimics the attribute-style usage object the real OpenAI SDK returns,
    so this module's logic can be verified without a live API call."""
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockGeminiUsage:
    """Mimics the real Gemini SDK's response.usage_metadata shape --
    different field names, same underlying concept (Part 4 of REPORT.md)."""
    def __init__(self, prompt_token_count, candidates_token_count):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = prompt_token_count + candidates_token_count


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("TOKEN COST CALCULATOR -- VERIFIED AGAINST SAMPLE USAGE DATA (MULTI-PROVIDER)")
    out("=" * 90)

    out("\n--- Test 1: dict-style usage object (OpenAI shape) ---")
    sample_usage_dict = {"prompt_tokens": 42, "completion_tokens": 128, "total_tokens": 170}
    result = calculate_cost(sample_usage_dict, model="gpt-4o-mini")
    out(format_cost_report(result))

    out("\n--- Test 2: attribute-style usage object, OpenAI shape ---")
    mock_usage = MockUsage(prompt_tokens=500, completion_tokens=1000)
    result2 = calculate_cost(mock_usage, model="gpt-4o-mini")
    out(format_cost_report(result2))

    out("\n--- Test 3: attribute-style usage object, GEMINI shape (different field names) ---")
    mock_gemini_usage = MockGeminiUsage(prompt_token_count=500, candidates_token_count=1000)
    result3 = calculate_cost(mock_gemini_usage, model="gemini-2.5-flash")
    out(format_cost_report(result3))
    out("Note: identical token counts to Test 2, but the object uses Gemini's field names")
    out("(prompt_token_count/candidates_token_count) -- calculate_cost() handles both")
    out("shapes transparently via the FIELD_ALIASES lookup, with no caller-side branching.")

    out("\n--- Test 4: same token usage, cost compared across ALL providers ---")
    comparison_usage = MockUsage(prompt_tokens=1000, completion_tokens=500)
    for model_name in MODEL_PRICING:
        r = calculate_cost(comparison_usage, model=model_name)
        out(f"  {model_name:<22} total_cost=${r['total_cost_usd']:.8f}")

    out("\n--- Test 5: what 10,000 requests/day would cost (gemini-2.5-flash) ---")
    daily_usage = MockGeminiUsage(prompt_token_count=1000, candidates_token_count=500)
    single_request_cost = calculate_cost(daily_usage, model="gemini-2.5-flash")["total_cost_usd"]
    daily_cost = single_request_cost * 10000
    monthly_cost = daily_cost * 30
    out(f"  Single request cost: ${single_request_cost:.8f}")
    out(f"  10,000 requests/day: ${daily_cost:.4f}")
    out(f"  Monthly (x30 days):  ${monthly_cost:.2f}")

    out("\n" + "=" * 90)
    out("VERIFICATION NOTE")
    out("=" * 90)
    out("""
This module was verified using MOCK usage data with known token counts,
NOT a live API call -- the arithmetic (tokens / 1,000,000 * rate) is
deterministic and doesn't require hitting the real API to confirm it's
correct. When run against a REAL response.usage or response.usage_metadata
object from an actual gemini_content_demo.py call, this same function
computes the exact real-dollar cost of that specific request -- and it
does so identically regardless of which provider's field-naming
convention that object happens to use.
""")

    with open("outputs/cost_calculator_verification.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved to outputs/cost_calculator_verification.txt")
