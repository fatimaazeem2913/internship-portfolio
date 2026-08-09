"""
function_calling_loop.py
----------------------------
Implements the complete function calling loop:
    1. Send tools + user question to the model
    2. Model decides to call a function (or not)
    3. WE execute that function locally (the model never runs code itself)
    4. We send the function's result back to the model
    5. Model synthesizes a final, natural-language answer using that result

This is manual (automatic_function_calling disabled) so every step is
visible and inspectable -- exactly what a real production system needs
to log, debug, and handle errors for.

SETUP:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python3 function_calling_loop.py
"""

import os
from google import genai
from google.genai import types
from tools import TOOL_REGISTRY, ALL_TOOLS

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant with access to tools. Use a tool whenever "
    "it would give you more accurate or current information than your own "
    "knowledge, especially for arithmetic, current time, or product data."
)


def run_function_calling_loop(user_question, verbose=True):
    """
    The full 5-step loop, manually implemented so every stage is visible.
    """
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_question)])]

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[ALL_TOOLS],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    if not response.function_calls:
        if verbose:
            print("[No function call requested -- model answered directly]")
        return response.text

    function_call = response.function_calls[0]
    function_name = function_call.name
    function_args = dict(function_call.args)

    if verbose:
        print(f"[Model requested: {function_name}({function_args})]")

    if function_name not in TOOL_REGISTRY:
        function_result = {"error": f"Unknown function '{function_name}' requested by model."}
    else:
        func, _ = TOOL_REGISTRY[function_name]
        try:
            function_result = func(**function_args)
        except TypeError as e:
            function_result = {"error": f"Invalid arguments for {function_name}: {e}"}

    if verbose:
        print(f"[Local execution result: {function_result}]")

    function_call_content = response.candidates[0].content
    function_response_part = types.Part.from_function_response(
        name=function_name,
        response=function_result,
    )
    function_response_content = types.Content(role="user", parts=[function_response_part])

    contents.extend([function_call_content, function_response_content])

    final_response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[ALL_TOOLS],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return final_response.text


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("FULL FUNCTION CALLING LOOP -- 5 STAGES, ALL VISIBLE")
    out("=" * 90)

    test_questions = [
        "What time is it right now in Tokyo?",
        "What's 4127 multiplied by 8912, divided by 3?",
        "Do you have a wireless keyboard in stock, and how much is it?",
        "Format 45000.5 as Pakistani Rupees.",
    ]

    import io
    import contextlib
    for q in test_questions:
        out(f"\n{'-'*90}")
        out(f"QUESTION: {q}")
        out("-" * 90)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            answer = run_function_calling_loop(q)
        out(buf.getvalue().strip())
        out(f"\nFINAL ANSWER: {answer}")

    with open("outputs/function_calling_loop_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/function_calling_loop_results.txt")
