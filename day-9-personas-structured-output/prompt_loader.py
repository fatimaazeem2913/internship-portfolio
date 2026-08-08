"""
prompt_loader.py
-------------------
Loads prompt templates from separate .md files (rather than hardcoding
them inside Python scripts), keeping prompt content cleanly separated
from application/API-calling logic -- Day 6's Best Practice #10.

Each .md file is structured as:

    # Title
    ...
    ---
    ## SECTION_NAME
    <prompt text, possibly with {placeholders}>
    ---
    ## ANOTHER_SECTION
    ...

This loader parses out each "## SECTION_NAME" block into a dict entry.
"""

import re


def load_prompts(md_path):
    """
    Parse a markdown prompt file into a dict of {section_name: prompt_text}.

    Args:
        md_path (str): path to the .md file.

    Returns:
        dict[str, str]: section name -> raw prompt template text (with
        {placeholder} syntax intact, ready for .format()).
    """
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"\n## ([A-Z_]+)\n", content)
    prompts = {}
    for i in range(1, len(sections), 2):
        name = sections[i].strip()
        body = sections[i + 1].split("\n---")[0].strip()
        prompts[name] = body

    return prompts


if __name__ == "__main__":
    production = load_prompts("prompts/production_prompts.md")
    expected_production = [
        "JSON_GENERATION_SYSTEM", "JSON_GENERATION_USER",
        "TEXT_PARSING_SYSTEM", "TEXT_PARSING_USER",
        "CODE_GENERATION_SYSTEM", "CODE_GENERATION_USER",
        "SUMMARIZATION_SYSTEM", "SUMMARIZATION_USER",
    ]
    for name in expected_production:
        assert name in production, f"Missing section: {name}"
    print(f"production_prompts.md loaded OK: {list(production.keys())}")

    personas = load_prompts("prompts/personas.md")
    expected_personas = ["PERSONA_FORMAL", "PERSONA_CASUAL", "PERSONA_TECHNICAL", "PERSONA_TEST_QUESTION"]
    for name in expected_personas:
        assert name in personas, f"Missing section: {name}"
    print(f"personas.md loaded OK: {list(personas.keys())}")

    print("\nAll prompt files loaded and validated successfully.")
