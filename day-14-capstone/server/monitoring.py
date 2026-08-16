"""
monitoring.py
----------------
Requirement #8: "For every LLM request, the application shall record the
timestamp, session ID, activity name, user prompt, input/output/total
token usage, Time to First Token (TTFT), and total response generation
time in a dedicated log file."

Extends Day 11's structured JSON logging pattern with the two new fields
this requirement specifically calls for: TTFT and activity name.

WHAT TTFT MEASURES, PRECISELY: the time from when a streaming request
starts until the FIRST chunk of generated text arrives -- exactly what
Day 8's streaming-vs-non-streaming finding was about (perceived
responsiveness), now a real, logged, measurable metric. TOTAL response
time measures the complete generation, start to finish -- TTFT will
always be <= total time.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "monitoring.log")

logger = logging.getLogger("capstone_monitoring")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.FileHandler(LOG_FILE)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)

    _stdout_handler = logging.StreamHandler()
    _stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_stdout_handler)


def log_llm_request(session_id, activity, user_prompt, prompt_tokens,
                     completion_tokens, total_tokens, ttft_ms, total_time_ms):
    """Logs one structured line per LLM request, with exactly the fields requirement #8 specifies."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "activity": activity,
        "user_prompt": user_prompt[:200],
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "total_response_time_ms": round(total_time_ms, 2),
    }
    logger.info(json.dumps(entry))
    return entry


class TTFTTimer:
    """
    A small helper for measuring TTFT and total time cleanly around a
    streaming call: start() when the request begins, mark_first_chunk()
    the moment the first piece of text arrives, and elapsed_ms() /
    ttft_ms() to read back both measurements once the stream completes.
    """
    def __init__(self):
        self._start = None
        self._first_chunk_time = None

    def start(self):
        self._start = time.perf_counter()

    def mark_first_chunk(self):
        if self._first_chunk_time is None:
            self._first_chunk_time = time.perf_counter()

    def ttft_ms(self):
        if self._first_chunk_time is None or self._start is None:
            return 0.0
        return (self._first_chunk_time - self._start) * 1000

    def elapsed_ms(self):
        if self._start is None:
            return 0.0
        return (time.perf_counter() - self._start) * 1000


if __name__ == "__main__":
    print("=" * 90)
    print("MONITORING -- SELF-TEST (pure Python, no API needed)")
    print("=" * 90)

    print(f"\nLog file location: {LOG_FILE}")

    timer = TTFTTimer()
    timer.start()
    time.sleep(0.05)
    timer.mark_first_chunk()
    time.sleep(0.1)

    entry = log_llm_request(
        session_id="test-session-123",
        activity="brain_buster",
        user_prompt="Can I have a hint please?",
        prompt_tokens=42,
        completion_tokens=18,
        total_tokens=60,
        ttft_ms=timer.ttft_ms(),
        total_time_ms=timer.elapsed_ms(),
    )

    print(f"\nLogged entry: {entry}")
    assert 40 < entry["ttft_ms"] < 100, f"TTFT should be ~50ms, got {entry['ttft_ms']}"
    assert entry["total_response_time_ms"] > entry["ttft_ms"], "Total time must exceed TTFT"
    print(f"\nVerified: TTFT ({entry['ttft_ms']}ms) < total time ({entry['total_response_time_ms']}ms) -- correct.")

    with open(LOG_FILE) as f:
        lines = f.readlines()
    print(f"\nLog file now contains {len(lines)} line(s). Last line is valid JSON: {bool(json.loads(lines[-1]))}")

    print("\nSelf-test passed.")
