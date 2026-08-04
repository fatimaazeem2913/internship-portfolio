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
    cot = load_prompts("prompts/cot_prompts.md")
    assert "DIRECT_PROMPT" in cot and "COT_PROMPT" in cot
    print(f"cot_prompts.md loaded OK: {list(cot.keys())}")

    shots = load_prompts("prompts/shot_prompts.md")
    expected = ["CLASSIFICATION_ZERO_SHOT", "CLASSIFICATION_ONE_SHOT", "CLASSIFICATION_FEW_SHOT",
                "EXTRACTION_ZERO_SHOT", "EXTRACTION_ONE_SHOT", "EXTRACTION_FEW_SHOT",
                "GENERATION_ZERO_SHOT", "GENERATION_ONE_SHOT", "GENERATION_FEW_SHOT"]
    for name in expected:
        assert name in shots, f"Missing section: {name}"
    print(f"shot_prompts.md loaded OK: {list(shots.keys())}")

    react = load_prompts("prompts/react_prompts.md")
    assert "REACT_SYSTEM_PROMPT" in react and "REACT_USER_QUESTION" in react
    print(f"react_prompts.md loaded OK: {list(react.keys())}")

    print("\nAll prompt files loaded and validated successfully.")
