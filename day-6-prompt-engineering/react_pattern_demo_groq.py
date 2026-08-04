"""
react_pattern_demo_groq.py
--------------------------------
FREE VERSION: uses Groq's free API tier instead of OpenAI for the real
agentic Thought -> Action -> Observation loop. Fully OpenAI-compatible SDK.

SETUP (run locally):
    pip install openai
    Get a free key at https://console.groq.com/keys (no credit card needed)
    export GROQ_API_KEY="gsk_...your-key-here..."
    python3 react_pattern_demo_groq.py
"""

import os
import re
from openai import OpenAI
from prompt_loader import load_prompts

MODEL = "llama-3.3-70b-versatile"
MAX_TURNS = 8  # hard turn limit -- a real production safeguard (Day 6 study guide, Part 4)

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

FAKE_KNOWLEDGE_BASE = {
    "population of japan": "approximately 123,000,000 (2026 estimate)",
    "population of germany": "approximately 83,500,000 (2026 estimate)",
    "gdp per capita japan": "approximately $34,000 USD (2026 estimate)",
    "gdp per capita germany": "approximately $52,000 USD (2026 estimate)",
}


def search_tool(query):
    query_lower = query.lower().strip()
    query_words = set(re.findall(r"[a-z]+", query_lower))
    best_match, best_score = None, 0
    for key, value in FAKE_KNOWLEDGE_BASE.items():
        key_words = set(re.findall(r"[a-z]+", key))
        overlap = len(query_words & key_words)
        if overlap > best_score:
            best_score, best_match = overlap, value
    if best_match and best_score >= 1:
        return best_match
    return f"No result found for '{query}'."
def calculator_tool(expression):
    try:
        allowed = set("0123456789.+-*/() ")
        if not set(expression) <= allowed:
            return "Error: invalid characters in expression."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
TOOLS = {"Search": search_tool, "Calculator": calculator_tool}
ACTION_PATTERN = re.compile(r"Action:\s*(\w+)\[(.*?)\]")
FINAL_PATTERN = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def call_openai(messages, temperature=0.0):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    prompts = load_prompts("prompts/react_prompts.md")
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out(f"REACT PATTERN -- REAL Groq Free API, REAL AGENTIC LOOP ({MODEL})")
    out("=" * 100)

    system_prompt = prompts["REACT_SYSTEM_PROMPT"]
    question = prompts["REACT_USER_QUESTION"]
    out(f"\nQuestion: {question}\n")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for turn in range(1, MAX_TURNS + 1):
        response_text = call_openai(messages)
        out(f"\n--- Turn {turn} ---")
        out(response_text)
        messages.append({"role": "assistant", "content": response_text})

        final_match = FINAL_PATTERN.search(response_text)
        if final_match:
            out("\n[Loop ended -- Final Answer received]")
            break

        action_match = ACTION_PATTERN.search(response_text)
        if action_match:
            tool_name, tool_input = action_match.group(1), action_match.group(2)
            if tool_name in TOOLS:
                result = TOOLS[tool_name](tool_input)
            else:
                result = f"Error: unknown tool '{tool_name}'"
            observation = f"Observation: {result}"
            out(observation)
            messages.append({"role": "user", "content": observation})
        else:
            out("\n[No Action or Final Answer detected -- stopping loop]")
            break
    else:
        out(f"\n[Hit MAX_TURNS={MAX_TURNS} safeguard -- stopping loop]")

    with open("outputs/react_groq_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n\nSaved to outputs/react_groq_results.txt")
