"""
tool_choice_modes_demo.py
-----------------------------
Studies the function calling specification's tool_choice parameter --
Gemini's equivalent is tool_config.function_calling_config.mode, with
three values directly analogous to OpenAI's tool_choice:

    AUTO  (~ OpenAI tool_choice="auto")    -- model decides whether to
          call a tool at all, based on the question. THE DEFAULT.
    ANY   (~ OpenAI tool_choice="required") -- model MUST call one of the
          provided tools, even if it wouldn't otherwise choose to.
    NONE  (~ OpenAI tool_choice="none")     -- tools are visible to the
          model but calling is DISABLED; the model must answer directly.

This script demonstrates all three modes against the SAME question, to
make the difference in behavior directly observable.

SETUP:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python3 tool_choice_modes_demo.py
"""

import os
from google import genai
from google.genai import types
from tools import ALL_TOOLS

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

TEST_QUESTION = "Tell me something interesting about the number 42."


def call_with_mode(mode):
    config_kwargs = {
        "tools": [ALL_TOOLS],
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if mode is not None:
        config_kwargs["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=mode)
        )

    response = client.models.generate_content(
        model=MODEL,
        contents=TEST_QUESTION,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("TOOL_CHOICE / TOOL_CONFIG MODES: AUTO vs ANY vs NONE")
    out("=" * 90)
    out(f"\nTest question (deliberately tool-use-optional): \"{TEST_QUESTION}\"\n")

    out("-" * 90)
    out("MODE: AUTO (default -- model decides for itself)")
    out("-" * 90)
    resp = call_with_mode("AUTO")
    if resp.function_calls:
        out(f"Model chose to call a function: {resp.function_calls[0].name}({dict(resp.function_calls[0].args)})")
    else:
        out(f"Model answered directly (no tool call): {resp.text}")

    out("\n" + "-" * 90)
    out("MODE: ANY (model is FORCED to call some tool, even if unhelpful)")
    out("-" * 90)
    resp = call_with_mode("ANY")
    if resp.function_calls:
        out(f"Model was forced to call: {resp.function_calls[0].name}({dict(resp.function_calls[0].args)})")
        out("(Notice: this tool call may not actually be useful for this")
        out(" question -- ANY mode forces SOME call, not necessarily a SENSIBLE one.)")
    else:
        out(f"Unexpected: no function call even in ANY mode: {resp.text}")

    out("\n" + "-" * 90)
    out("MODE: NONE (tools visible but calling disabled -- must answer directly)")
    out("-" * 90)
    resp = call_with_mode("NONE")
    if resp.function_calls:
        out(f"Unexpected: function call occurred despite NONE mode: {resp.function_calls}")
    else:
        out(f"Model answered directly, as required: {resp.text}")

    out("\n" + "=" * 90)
    out("WHEN TO USE EACH MODE IN PRODUCTION")
    out("=" * 90)
    out("""
AUTO (the default): use for general-purpose assistants where you genuinely
  don't know in advance whether a given user question will need a tool --
  let the model decide per-request, exactly like function_calling_loop.py
  and multi_tool_agent.py in this project.

ANY: use when you KNOW the task always requires a tool call and want to
  eliminate the "model declines" edge case entirely -- e.g. a dedicated
  "calculate my order total" endpoint that should ALWAYS invoke your
  pricing tools, never attempt to answer from memory. Trade-off: if the
  question genuinely doesn't need a tool, ANY mode still forces one,
  potentially producing an unhelpful or irrelevant call.

NONE: use to temporarily disable tool use without removing the tool
  definitions from your code -- useful for A/B testing tool-augmented vs.
  non-tool-augmented responses, or for a "safe mode" fallback if a tool
  backend is known to be down and you'd rather get a direct (possibly
  less accurate) answer than a guaranteed failed tool call.
""")

    with open("outputs/tool_choice_modes_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/tool_choice_modes_results.txt")
