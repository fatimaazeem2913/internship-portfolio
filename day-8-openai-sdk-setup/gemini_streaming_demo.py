"""
gemini_streaming_demo.py
----------------------------
Implements streaming responses (generate_content_stream) and times it
against a standard non-streaming call to the same prompt, to compare the
actual user experience difference: time-to-first-token vs. total
completion time.

SETUP: same as gemini_content_demo.py -- needs GEMINI_API_KEY set.
    python3 gemini_streaming_demo.py
"""

import os
import time
from google import genai
from google.genai import types
from token_cost_calculator import calculate_cost, format_cost_report

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = "You are a helpful technical writer."
USER_PROMPT = "Write a 150-word explanation of how DNS resolution works."


def run_non_streaming():
    """Standard call: wait for the ENTIRE response before anything is returned."""
    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=USER_PROMPT,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
        ),
    )
    total_time = time.perf_counter() - start

    return {
        "text": response.text,
        "time_to_first_content": total_time,
        "total_time": total_time,
        "usage": response.usage_metadata,
    }


def run_streaming():
    """
    Streaming call: the response arrives as a sequence of chunks. We can
    measure time-to-FIRST-chunk separately from time-to-COMPLETE, which is
    the actual UX difference streaming provides.
    """
    start = time.perf_counter()
    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=USER_PROMPT,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
        ),
    )

    first_chunk_time = None
    full_text = ""
    usage = None

    for chunk in stream:
        if chunk.text:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start
            full_text += chunk.text
        if chunk.usage_metadata:
            usage = chunk.usage_metadata  # final chunk carries the complete usage totals

    total_time = time.perf_counter() - start

    return {
        "text": full_text,
        "time_to_first_content": first_chunk_time,
        "total_time": total_time,
        "usage": usage,
    }


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out(f"STREAMING vs NON-STREAMING COMPARISON ({MODEL})")
    out("=" * 90)

    out("\n--- NON-STREAMING (standard generate_content call) ---")
    non_stream_result = run_non_streaming()
    out(f"Time until ANY output is visible: {non_stream_result['time_to_first_content']:.3f}s")
    out(f"Total time:                        {non_stream_result['total_time']:.3f}s")
    out(f"Response text:\n{non_stream_result['text']}")

    out("\n--- STREAMING (generate_content_stream) ---")
    stream_result = run_streaming()
    out(f"Time until FIRST token is visible: {stream_result['time_to_first_content']:.3f}s")
    out(f"Total time (all tokens received):  {stream_result['total_time']:.3f}s")
    out(f"Response text:\n{stream_result['text']}")

    out("\n" + "=" * 90)
    out("THE ACTUAL UX DIFFERENCE")
    out("=" * 90)
    speedup = non_stream_result['time_to_first_content'] / stream_result['time_to_first_content']
    out(f"""
Non-streaming: the user sees NOTHING for the full {non_stream_result['time_to_first_content']:.2f}s
duration of the request, then the ENTIRE response appears at once.

Streaming: the user sees the FIRST token after only {stream_result['time_to_first_content']:.2f}s
-- a {speedup:.1f}x improvement in PERCEIVED responsiveness -- even though
TOTAL completion time is roughly the same in both cases (streaming doesn't
make the model generate faster, Day 5's autoregressive mechanism still
applies token-by-token). Streaming improves perceived latency, not actual
total generation time -- this is precisely why every production chat UI
(ChatGPT, Claude, Gemini) streams responses rather than waiting for
completion.
""")

    if stream_result["usage"]:
        cost = calculate_cost(stream_result["usage"], model=MODEL)
        out("--- COST (identical for streaming and non-streaming -- same tokens either way) ---")
        out(format_cost_report(cost))

    with open("outputs/gemini_streaming_comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved to outputs/gemini_streaming_comparison_results.txt")
