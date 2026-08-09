"""
tools.py
-----------
Defines 4 custom tools with proper JSON schemas for function calling:
    1. get_current_time  -- returns the current time in a given timezone
    2. calculate          -- evaluates a safe arithmetic expression
    3. search_database     -- mock lookup against a small in-memory dataset
    4. format_currency      -- formats a number as a currency string

Each tool has TWO parts, deliberately kept separate:
    (a) the actual Python function -- pure, no API dependency, fully
        testable and verifiable on its own
    (b) a types.FunctionDeclaration -- the JSON-schema description that
        gets sent to the model so it knows the tool exists, what it does,
        and what arguments it needs

This separation matters: the SCHEMA is what the model sees and reasons
about; the FUNCTION is what your own code actually executes once the
model decides to call it. They must stay in sync, but they are not the
same artifact.
"""

from datetime import datetime, timezone, timedelta
from google.genai import types

# ============================================================
# TOOL 1: get_current_time
# ============================================================

TIMEZONE_OFFSETS = {
    "UTC": 0,
    "EST": -5, "America/New_York": -5,
    "PST": -8, "America/Los_Angeles": -8,
    "GMT": 0, "Europe/London": 0,
    "CET": 1, "Europe/Paris": 1,
    "PKT": 5, "Asia/Karachi": 5,
    "IST": 5.5, "Asia/Kolkata": 5.5,
    "JST": 9, "Asia/Tokyo": 9,
}


def get_current_time(timezone_name: str) -> dict:
    """
    Returns the current date and time in the specified timezone.

    Args:
        timezone_name: a timezone identifier, e.g. "UTC", "PKT",
            "America/New_York", "Asia/Tokyo".

    Returns:
        dict with keys: timezone, datetime (ISO format), day_of_week.
        On an unrecognized timezone, returns an "error" key instead --
        this is a deliberate, real error path (Part 5's edge case).
    """
    if timezone_name not in TIMEZONE_OFFSETS:
        return {
            "error": f"Unknown timezone '{timezone_name}'. "
                     f"Known timezones: {sorted(TIMEZONE_OFFSETS.keys())}"
        }

    offset_hours = TIMEZONE_OFFSETS[timezone_name]
    now_utc = datetime.now(timezone.utc)
    local_time = now_utc + timedelta(hours=offset_hours)

    return {
        "timezone": timezone_name,
        "datetime": local_time.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week": local_time.strftime("%A"),
    }


GET_CURRENT_TIME_DECLARATION = types.FunctionDeclaration(
    name="get_current_time",
    description="Get the current date and time in a specified timezone.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "timezone_name": types.Schema(
                type="STRING",
                description="Timezone identifier, e.g. 'UTC', 'PKT', 'America/New_York', 'Asia/Tokyo'.",
            ),
        },
        required=["timezone_name"],
    ),
)


# ============================================================
# TOOL 2: calculate
# ============================================================

def calculate(expression: str) -> dict:
    """
    Safely evaluates a basic arithmetic expression.

    Args:
        expression: a string containing only digits, and the operators
            + - * / ( ) . and spaces -- e.g. "123000000 * 34000".

    Returns:
        dict with "result" on success, or "error" on invalid input --
        NEVER raises an exception, since this function's job is to hand
        back a clean result the model can synthesize a response from,
        not to crash the calling loop (Part 5's edge case: functions
        should fail gracefully, not raise).
    """
    allowed_chars = set("0123456789.+-*/() ")
    if not expression or not set(expression) <= allowed_chars:
        return {"error": f"Invalid characters in expression: '{expression}'. "
                          f"Only digits and + - * / ( ) . are allowed."}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except ZeroDivisionError:
        return {"error": "Division by zero."}
    except Exception as e:
        return {"error": f"Could not evaluate expression: {e}"}


CALCULATE_DECLARATION = types.FunctionDeclaration(
    name="calculate",
    description="Evaluate a basic arithmetic expression (addition, subtraction, "
                "multiplication, division, parentheses). Always use this instead "
                "of computing arithmetic mentally, especially for large numbers.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "expression": types.Schema(
                type="STRING",
                description="A valid arithmetic expression, e.g. '123000000 * 34000' or '(50 + 25) / 3'.",
            ),
        },
        required=["expression"],
    ),
)


# ============================================================
# TOOL 3: search_database (mock)
# ============================================================

MOCK_PRODUCT_DATABASE = {
    "wireless keyboard": {"price_usd": 79.99, "stock": 142, "category": "peripherals"},
    "usb-c hub": {"price_usd": 34.50, "stock": 8, "category": "peripherals"},
    "laptop stand": {"price_usd": 45.00, "stock": 0, "category": "accessories"},
    "wireless mouse": {"price_usd": 29.99, "stock": 310, "category": "peripherals"},
    "monitor 27 inch": {"price_usd": 289.99, "stock": 22, "category": "displays"},
}


def search_database(query: str) -> dict:
    """
    Mock product database lookup. Uses simple keyword-overlap matching
    (the same real fix applied to Day 6's ReAct search tool after
    discovering exact-match lookups fail on natural model phrasing) --
    NOT exact string matching, since a real model's query wording won't
    always match a database key exactly.

    Args:
        query: a natural-language product search query.

    Returns:
        dict with "found": bool, and either the matched product's data
        or a "message" explaining no match was found.
    """
    query_words = set(query.lower().split())
    best_match, best_score = None, 0

    for product_name, data in MOCK_PRODUCT_DATABASE.items():
        product_words = set(product_name.split())
        overlap = len(query_words & product_words)
        if overlap > best_score:
            best_score, best_match = overlap, product_name

    if best_match:
        return {"found": True, "product": best_match, **MOCK_PRODUCT_DATABASE[best_match]}
    return {"found": False, "message": f"No product matching '{query}' found in the database."}


SEARCH_DATABASE_DECLARATION = types.FunctionDeclaration(
    name="search_database",
    description="Search the mock product database by natural-language query. "
                "Returns price, stock level, and category if a match is found.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "query": types.Schema(
                type="STRING",
                description="Natural-language search query, e.g. 'wireless keyboard' or 'usb hub'.",
            ),
        },
        required=["query"],
    ),
)


# ============================================================
# TOOL 4: format_currency
# ============================================================

CURRENCY_SYMBOLS = {
    "USD": ("$", 2), "EUR": ("\u20ac", 2), "GBP": ("\u00a3", 2),
    "PKR": ("Rs", 0), "JPY": ("\u00a5", 0), "INR": ("\u20b9", 2),
}


def format_currency(amount, currency_code: str) -> dict:
    """
    Formats a numeric amount as a currency string.

    Args:
        amount: the numeric amount to format (int or float).
        currency_code: a 3-letter currency code, e.g. "USD", "PKR", "JPY".

    Returns:
        dict with "formatted" on success, or "error" for an unknown
        currency code OR a non-numeric amount. JPY and PKR are
        deliberately formatted with 0 decimal places (a real convention
        -- neither currency uses subunits in everyday pricing),
        demonstrating the function isn't just naively appending a symbol.

    NOTE ON A REAL BUG FOUND DURING TESTING: an earlier version of this
    function type-hinted `amount: float` but did not validate it at
    runtime -- Python type hints are NOT enforced automatically. Calling
    format_currency(amount="a lot", ...) raised an uncaught ValueError
    from the f-string formatting itself ("Unknown format code 'f' for
    object of type 'str'"), rather than failing gracefully. Since a
    model-generated function call is exactly the kind of caller that can
    plausibly pass a wrong-typed argument, this function now validates
    amount's type explicitly and returns a clean {"error": ...} instead
    of crashing -- the same defense-in-depth principle from Day 9's JSON
    schema validation, applied here to function arguments instead of
    LLM-generated JSON output.
    """
    if not isinstance(amount, (int, float)):
        return {"error": f"'amount' must be a number, got {type(amount).__name__}: {amount!r}"}

    if currency_code not in CURRENCY_SYMBOLS:
        return {"error": f"Unknown currency code '{currency_code}'. "
                          f"Known codes: {sorted(CURRENCY_SYMBOLS.keys())}"}

    symbol, decimals = CURRENCY_SYMBOLS[currency_code]
    formatted_number = f"{amount:,.{decimals}f}"
    return {"formatted": f"{symbol}{formatted_number}", "currency_code": currency_code}


FORMAT_CURRENCY_DECLARATION = types.FunctionDeclaration(
    name="format_currency",
    description="Format a numeric amount as a properly formatted currency string "
                "for the given 3-letter currency code (USD, EUR, GBP, PKR, JPY, INR).",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "amount": types.Schema(type="NUMBER", description="The numeric amount to format."),
            "currency_code": types.Schema(type="STRING", description="3-letter currency code, e.g. 'USD'."),
        },
        required=["amount", "currency_code"],
    ),
)


# ============================================================
# Registry: maps function name -> (callable, declaration)
# ============================================================
TOOL_REGISTRY = {
    "get_current_time": (get_current_time, GET_CURRENT_TIME_DECLARATION),
    "calculate": (calculate, CALCULATE_DECLARATION),
    "search_database": (search_database, SEARCH_DATABASE_DECLARATION),
    "format_currency": (format_currency, FORMAT_CURRENCY_DECLARATION),
}

ALL_DECLARATIONS = [decl for _, decl in TOOL_REGISTRY.values()]
ALL_TOOLS = types.Tool(function_declarations=ALL_DECLARATIONS)


if __name__ == "__main__":
    print("=" * 90)
    print("TOOL FUNCTIONS -- SELF-TEST (pure Python, no API needed)")
    print("=" * 90)

    print("\n--- get_current_time ---")
    print(get_current_time("PKT"))
    print(get_current_time("Asia/Tokyo"))
    print(get_current_time("Mars/OlympusMons"))  # deliberate error case

    print("\n--- calculate ---")
    print(calculate("123000000 * 34000"))
    print(calculate("(50 + 25) / 5"))
    print(calculate("10 / 0"))              # deliberate error case
    print(calculate("import os; os.system('ls')"))  # deliberate injection attempt

    print("\n--- search_database ---")
    print(search_database("wireless keyboard"))
    print(search_database("USB hub for laptop"))    # partial/natural phrasing
    print(search_database("bluetooth speaker"))      # deliberate no-match case

    print("\n--- format_currency ---")
    print(format_currency(1234.5, "USD"))
    print(format_currency(50000, "PKR"))
    print(format_currency(999.99, "JPY"))
    print(format_currency(100, "XYZ"))       # deliberate error case

    print("\nAll 4 tools self-tested. Registered tool names:", list(TOOL_REGISTRY.keys()))
