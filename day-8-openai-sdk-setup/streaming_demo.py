"""
streaming_demo.py
---------------------
Implements streaming responses (stream=True) and times it against a
standard non-streaming call to the same prompt, to compare the actual
user experience difference: time-to-first-token vs. total completion time.

SETUP: same as chat_completions_demo.py -- needs OPENAI_API_KEY set.
    python3 streaming_demo.py
"""

import time
from openai import OpenAI
from token_cost_calculator import calculate_cost, format_cost_report

client = OpenAI()
MODEL = "gpt-4o-mini"

PROMPT = [
    {"role": "system", "content": "You are a helpful technical writer."},
    {"role": "user", "content": "Write a 150-word explanation of how DNS resolution works."},
]


def run_non_streaming():
    """Standard call: wait for the ENTIRE response before anything is returned."""
    start = time.perf_counter()
    response = client.chat.completions.create(model=MODEL, messages=PROMPT, temperature=0.3)
    total_time = time.perf_counter() - start

    return {
        "text": response.choices[0].message.content,
        "time_to_first_content": total_time,
        "total_time": total_time,
        "usage": response.usage,
    }


def run_streaming():
    """
    Streaming call: the response arrives as a sequence of chunks. We can
    measure time-to-FIRST-chunk separately from time-to-COMPLETE, which is
    the actual UX difference streaming provides.
    """
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL, messages=PROMPT, temperature=0.3, stream=True,
        stream_options={"include_usage": True},
    )

    first_chunk_time = None
    full_text = ""
    usage = None

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start
            full_text += chunk.choices[0].delta.content
        if chunk.usage:
            usage = chunk.usage

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

    out("\n--- NON-STREAMING (stream=False, the default) ---")
    non_stream_result = run_non_streaming()
    out(f"Time until ANY output is visible: {non_stream_result['time_to_first_content']:.3f}s")
    out(f"Total time:                        {non_stream_result['total_time']:.3f}s")
    out(f"Response text:\n{non_stream_result['text']}")

    out("\n--- STREAMING (stream=True) ---")
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
(ChatGPT, Claude, etc.) streams responses rather than waiting for
completion.
""")

    if stream_result["usage"]:
        cost = calculate_cost(stream_result["usage"], model=MODEL)
        out("--- COST (identical for streaming and non-streaming -- same tokens either way) ---")
        out(format_cost_report(cost))

    with open("outputs/streaming_comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved to outputs/streaming_comparison_results.txt")
