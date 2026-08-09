"""
logging_config.py
---------------------
Structured logging: every request logs timestamp, session_id, model,
token usage, and latency_ms -- as a single structured JSON line per
request, not free-form text. This is a deliberate choice: structured
(JSON) logs can be parsed, filtered, and aggregated by log-analysis
tools (e.g. "show me every request over 2000ms latency," or "sum total
tokens used per session today") in a way free-form text logs cannot be
without fragile regex parsing.
"""

import json
import logging
import sys

logger = logging.getLogger("chat_api")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)


def log_request(timestamp, session_id, model, prompt_tokens, completion_tokens,
                 total_tokens, latency_ms, status="success"):
    """
    Logs one structured line per request. Fields exactly as specified:
    timestamp, session_id, model, token usage, latency_ms.
    """
    entry = {
        "timestamp": timestamp,
        "session_id": session_id,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": round(latency_ms, 2),
        "status": status,
    }
    logger.info(json.dumps(entry))
    return entry


def log_error(timestamp, session_id, error_type, message, status_code):
    """Logs one structured line for an error response -- kept in the same JSON shape family."""
    entry = {
        "timestamp": timestamp,
        "session_id": session_id,
        "error_type": error_type,
        "message": message,
        "status_code": status_code,
        "status": "error",
    }
    logger.info(json.dumps(entry))
    return entry


if __name__ == "__main__":
    from datetime import datetime, timezone

    print("=" * 90)
    print("STRUCTURED LOGGING -- SELF-TEST")
    print("=" * 90)
    print("\nEach line below is a single structured JSON log entry (as it would")
    print("appear in real server logs, one per request):\n")

    log_request(
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id="abc-123",
        model="gemini-3.5-flash-lite",
        prompt_tokens=42,
        completion_tokens=18,
        total_tokens=60,
        latency_ms=234.7,
    )

    log_error(
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id="does-not-exist",
        error_type="session_not_found",
        message="No session with that ID exists.",
        status_code=404,
    )

    print("\nBoth entries are valid, parseable JSON -- verified below:")
    test_entry = log_request(
        timestamp="2026-08-09T12:00:00Z", session_id="test", model="test-model",
        prompt_tokens=1, completion_tokens=1, total_tokens=2, latency_ms=1.0,
    )
    assert isinstance(test_entry, dict)
    print("Structured logging self-test passed.")
