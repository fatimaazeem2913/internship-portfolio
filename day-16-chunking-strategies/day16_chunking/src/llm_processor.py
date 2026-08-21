import json
from typing import Dict, Any

def generate_table_enrichment_prompt(raw_table_text: str) -> str:
    return f"""Analyze the extracted raw table text:
---
{raw_table_text}
---
Generate a strict JSON response containing:
1. "markdown_table": Valid Markdown formatted table.
2. "table_summary": A high-level 2-sentence summary.
3. "structured_json": Array of JSON key-value objects for direct retrieval lookup.
"""

def post_process_table_mock(raw_text: str) -> Dict[str, Any]:
    """Simulated LLM pipeline for offline testing."""
    return {
        "status": "success",
        "markdown_table": "| Metric | Target | Measurement Method |\n|---|---|---|\n| API p95 | <400ms | Synthetic |",
        "table_summary": "Summary of system SLA and performance targets.",
        "structured_json": [
            {"metric": "API response time (p95)", "target": "< 400ms"}
        ]
    }
