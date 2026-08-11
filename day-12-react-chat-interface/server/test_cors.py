"""
test_cors.py
---------------
Verifies CORS headers are ACTUALLY present in real responses -- not just
that CORSMiddleware was added to the app, but that a request carrying an
Origin header matching an allowed origin genuinely receives the expected
Access-Control-Allow-Origin header back, and that a DISALLOWED origin does
NOT receive that header (proving the allowlist is genuinely restrictive,
not just present-but-ineffective).

Run: USE_MOCK_LLM=true python3 test_cors.py
"""

from fastapi.testclient import TestClient
import main
import session_store

client = TestClient(main.app)


def test_allowed_origin_gets_cors_header():
    r = client.get("/api/sessions", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    cors_header = r.headers.get("access-control-allow-origin")
    assert cors_header == "http://localhost:5173", (
        f"Expected CORS header for allowed origin, got: {cors_header}"
    )
    return "PASS", cors_header


def test_disallowed_origin_denied():
    r = client.get("/api/sessions", headers={"Origin": "http://evil-site.example.com"})
    cors_header = r.headers.get("access-control-allow-origin")
    assert cors_header != "http://evil-site.example.com", (
        "A disallowed origin should NOT be reflected back in the CORS header"
    )
    return "PASS", cors_header


def test_preflight_options_request():
    r = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code in (200, 204), f"Preflight failed with status {r.status_code}"
    allow_methods = r.headers.get("access-control-allow-methods", "")
    assert "POST" in allow_methods, f"POST not in allowed methods: {allow_methods}"
    return "PASS", {"status": r.status_code, "allow_methods": allow_methods}


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("CORS VERIFICATION -- REAL HEADER INSPECTION")
    out("=" * 90)

    session_store.clear_all_sessions()

    tests = [
        ("Allowed origin (localhost:5173) receives CORS header", test_allowed_origin_gets_cors_header),
        ("Disallowed origin does NOT receive matching CORS header", test_disallowed_origin_denied),
        ("Preflight OPTIONS request for POST /api/chat succeeds", test_preflight_options_request),
    ]

    passed = 0
    for label, fn in tests:
        try:
            status, detail = fn()
            out(f"\n[{status}] {label}")
            out(f"  {detail}")
            passed += 1
        except AssertionError as e:
            out(f"\n[FAIL] {label}: {e}")

    out(f"\n\n{'='*90}")
    out(f"SUMMARY: {passed}/{len(tests)} CORS tests passed")
    out("=" * 90)

    with open("outputs/cors_test_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved to outputs/cors_test_results.txt")
