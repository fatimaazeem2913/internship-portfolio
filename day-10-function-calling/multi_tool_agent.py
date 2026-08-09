"""
multi_tool_agent.py
-----------------------
Builds a multi-tool agent that chains MULTIPLE function calls to answer a
single complex question requiring both DATA LOOKUP and CALCULATION --
neither tool alone can answer it.

THE QUESTION: "If I buy 3 wireless keyboards and 2 wireless mice, what's
the total cost in PKR?"

This genuinely requires several separate tool calls in sequence:
    1. search_database("wireless keyboard")  -> get price
    2. search_database("wireless mouse")      -> get price
    3. calculate("(3 * price1) + (2 * price2)")  -> get total (in USD)
    4. format_currency(total, "PKR")           -> final formatted answer

This mirrors Day 6's ReAct pattern (Thought -> Action -> Observation,
repeated) but expressed through native function calling instead of a
prompted text protocol -- the model decides each next tool call based on
what it has learned from previous results, in a real loop, not a
pre-scripted sequence.

SETUP:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python3 multi_tool_agent.py
"""

import os
from google import genai
from google.genai import types
from tools import TOOL_REGISTRY, ALL_TOOLS

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"
MAX_TOOL_CALLS = 8  # a real production safeguard against runaway loops (Day 6's ReAct lesson)

SYSTEM_INSTRUCTION = (
    "You are a shopping assistant with access to tools. For questions "
    "requiring both product data and calculation, use search_database to "
    "find prices FIRST, then use calculate for any arithmetic, then use "
    "format_currency if a specific currency format is requested. Never "
    "guess a price or compute large numbers mentally -- always use the "
    "appropriate tool."
)


def run_multi_tool_agent(user_question):
    """
    A genuine multi-turn agentic loop: keeps calling tools and feeding
    results back until the model has enough information to answer, up to
    MAX_TOOL_CALLS turns (Day 6's ReAct-style safeguard against a model
    getting stuck in a repetitive tool-calling loop).
    """
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_question)])]
    trace = []

    for turn in range(1, MAX_TOOL_CALLS + 1):
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
            trace.append({"turn": turn, "type": "final_answer", "text": response.text})
            return response.text, trace

        function_call = response.function_calls[0]
        function_name = function_call.name
        function_args = dict(function_call.args)

        if function_name not in TOOL_REGISTRY:
            function_result = {"error": f"Unknown function '{function_name}'"}
        else:
            func, _ = TOOL_REGISTRY[function_name]
            try:
                function_result = func(**function_args)
            except TypeError as e:
                function_result = {"error": f"Invalid arguments: {e}"}

        trace.append({
            "turn": turn, "type": "tool_call",
            "function": function_name, "args": function_args, "result": function_result,
        })

        function_call_content = response.candidates[0].content
        function_response_part = types.Part.from_function_response(
            name=function_name, response=function_result,
        )
        function_response_content = types.Content(role="user", parts=[function_response_part])
        contents.extend([function_call_content, function_response_content])

    trace.append({"turn": MAX_TOOL_CALLS + 1, "type": "max_calls_reached"})
    return None, trace


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("MULTI-TOOL AGENT -- CHAINING 2+ FUNCTION CALLS")
    out("=" * 90)

    question = "If I buy 3 wireless keyboards and 2 wireless mice, what's the total cost in PKR?"
    out(f"\nQuestion: {question}")
    out("\nThis requires: 2 database lookups (prices) + 1 calculation (total) ")
    out("+ 1 currency format conversion -- neither a single lookup nor a single")
    out("calculation alone can answer it.\n")

    final_answer, trace = run_multi_tool_agent(question)

    for step in trace:
        if step["type"] == "tool_call":
            out(f"--- Turn {step['turn']}: TOOL CALL ---")
            out(f"  Function: {step['function']}({step['args']})")
            out(f"  Result: {step['result']}")
        elif step["type"] == "final_answer":
            out(f"--- Turn {step['turn']}: FINAL ANSWER ---")
            out(f"  {step['text']}")
        elif step["type"] == "max_calls_reached":
            out(f"--- Hit MAX_TOOL_CALLS={MAX_TOOL_CALLS} safeguard ---")

    out(f"\n{'='*90}")
    out("SUMMARY")
    out("=" * 90)
    tool_calls = [s for s in trace if s["type"] == "tool_call"]
    out(f"Total tool calls chained: {len(tool_calls)}")
    out(f"Functions used, in order: {[s['function'] for s in tool_calls]}")
    out(f"\nFinal answer: {final_answer}")

    with open("outputs/multi_tool_agent_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/multi_tool_agent_results.txt")
