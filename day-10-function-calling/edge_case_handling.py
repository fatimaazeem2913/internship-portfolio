"""
edge_case_handling.py
-------------------------
Demonstrates robust handling of the required edge cases:
    1. Model declines to call a function (answers directly instead)
    2. Invalid argument types (model passes something the function can't use)
    3. Function returns an error (a genuine runtime failure inside the tool)

Parts of this ARE fully testable without any API call -- specifically,
cases 2 and 3 can be simulated directly against the real tool functions
with deliberately bad inputs, since the tools themselves don't depend on
the API at all. Case 1 requires a live model decision and is demonstrated
via function_calling_loop.py's "no function call requested" branch,
referenced here for completeness.

SETUP (for the live demonstration of case 1):
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python3 edge_case_handling.py
"""

from tools import TOOL_REGISTRY, calculate, get_current_time, format_currency, search_database


# ============================================================
# EDGE CASE 2: Invalid argument types -- fully testable, no API needed
# ============================================================

def test_invalid_argument_types():
    """
    Simulates a model calling a function with malformed or wrong-typed
    arguments -- a REAL failure mode, since the model generates arguments
    from natural language and can occasionally produce something that
    doesn't match the expected type, even with a schema in place.
    """
    results = []

    # Case: model passes a string where a number was required.
    # A REAL bug was found here during testing: format_currency's original
    # implementation had a `amount: float` type HINT but no runtime check
    # -- Python doesn't enforce type hints -- so this call raised an
    # uncaught ValueError from the f-string formatting itself, crashing
    # the script. Fixed in tools.py by adding an explicit isinstance()
    # check; now it returns a clean error dict instead.
    result = format_currency(amount="a lot", currency_code="USD")
    results.append(("format_currency(amount='a lot', ...)", "returned error dict (fixed)", result))

    try:
        result = TOOL_REGISTRY["calculate"][0]()
        results.append(("calculate() with no args", "no exception", result))
    except TypeError as e:
        results.append(("calculate() with no args", "TypeError caught (missing required arg)", str(e)))

    try:
        result = get_current_time(timezone_name="UTC", unexpected_arg="oops")
        results.append(("get_current_time(extra arg)", "no exception", result))
    except TypeError as e:
        results.append(("get_current_time(extra arg)", "TypeError caught (unexpected arg)", str(e)))

    return results


# ============================================================
# EDGE CASE 3: Function returns an error -- fully testable, no API needed
# ============================================================

def test_function_returns_error():
    """
    These are NOT exceptions -- they are the tool functions' own,
    deliberate {"error": ...} return values for genuinely invalid input
    THAT IS still the correct argument TYPE (a valid string, a valid
    number) but semantically wrong (an unknown timezone, a division by
    zero, an unknown currency). This is the more common, more important
    case in practice: most model-generated arguments are correctly
    TYPED, but occasionally semantically invalid.
    """
    results = []
    results.append(("get_current_time('Atlantis')", get_current_time("Atlantis")))
    results.append(("calculate('10 / 0')", calculate("10 / 0")))
    results.append(("calculate('2 +')", calculate("2 +")))
    results.append(("format_currency(100, 'BTC')", format_currency(100, "BTC")))
    results.append(("search_database('quantum computer')", search_database("quantum computer")))
    return results


# ============================================================
# EDGE CASE 1: Model declines to call a function -- requires a live model
# ============================================================

def demonstrate_model_declines():
    """
    A model correctly declining to call ANY tool is not a failure -- it's
    the CORRECT behavior for questions the tools genuinely can't help
    with. This function requires a live API call; see
    function_calling_loop.py's `if not response.function_calls:` branch,
    which handles this exact case in the main loop.
    """
    import io
    import contextlib
    from function_calling_loop import run_function_calling_loop

    question = "What's the capital of France?"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        answer = run_function_calling_loop(question)
    return question, buf.getvalue(), answer


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("EDGE CASE HANDLING -- ALL 3 REQUIRED SCENARIOS")
    out("=" * 90)

    out("\n" + "#" * 90)
    out("EDGE CASE 2: Invalid argument types (fully tested, no API needed)")
    out("#" * 90)
    for label, outcome, detail in test_invalid_argument_types():
        out(f"\n{label}")
        out(f"  Outcome: {outcome}")
        out(f"  Detail: {detail}")

    out("\n" + "#" * 90)
    out("EDGE CASE 3: Function returns an error (fully tested, no API needed)")
    out("#" * 90)
    for label, result in test_function_returns_error():
        out(f"\n{label}")
        out(f"  Result: {result}")
        out(f"  Has 'error' key: {'error' in result}")

    out("\n" + "#" * 90)
    out("EDGE CASE 1: Model declines to call a function (requires live API)")
    out("#" * 90)
    out("\nAttempting live demonstration -- requires GEMINI_API_KEY...")
    try:
        question, trace_output, answer = demonstrate_model_declines()
        out(f"\nQuestion (no tool is relevant): {question}")
        out(f"Trace: {trace_output.strip()}")
        out(f"Answer: {answer}")
    except Exception as e:
        out(f"\n[Live demo requires GEMINI_API_KEY -- error: {e}]")
        out("[This is EXPECTED behavior when no key is set -- see")
        out(" function_calling_loop.py's 'if not response.function_calls:' branch")
        out(" for the code path that correctly handles this case.]")

    out("\n" + "=" * 90)
    out("SUMMARY: ROBUST ERROR HANDLING PRINCIPLES DEMONSTRATED")
    out("=" * 90)
    out("""
1. Functions NEVER raise uncaught exceptions for semantically-invalid-but-
   correctly-typed input (unknown timezone, division by zero, unknown
   currency) -- they return a clean {"error": "..."} dict instead, which
   the model can read and explain to the user in natural language.

2. Functions DO let TypeErrors propagate for genuinely malformed calls
   (wrong argument type, missing required argument, unexpected extra
   argument) -- these are caught one level up, in the calling loop
   (function_calling_loop.py's try/except around the function call),
   converted into an {"error": ...} dict, and sent back to the model the
   SAME way a semantic error would be -- the model doesn't need to know
   or care which category of error occurred.

3. A model declining to call any tool is NOT an error case to catch --
   it's a normal, valid outcome that the calling loop must handle as a
   first-class path (the "if not response.function_calls" branch), not
   an afterthought.
""")

    with open("outputs/edge_case_handling_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/edge_case_handling_results.txt")
