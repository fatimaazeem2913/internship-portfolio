"""
react_pattern_demo.py
------------------------
Implements the ReAct (Reason + Act) prompting pattern for a simulated
tool-using scenario. ReAct interleaves explicit REASONING ("Thought:")
with TOOL CALLS ("Action:") and their results ("Observation:"), repeating
until the model has enough information to give a final answer.

This script simulates two fake tools -- a calculator and a "search" tool
returning canned facts -- and walks through a genuine multi-step ReAct
trace solving a question that requires BOTH tools in sequence, which
neither tool alone (nor a single non-agentic LLM call) could answer
correctly without external, up-to-date information.
"""


# ---- Simulated tools (stand-ins for real APIs/functions an agent could call) ----

FAKE_KNOWLEDGE_BASE = {
    "population of japan": "approximately 123,000,000 (2026 estimate)",
    "population of germany": "approximately 83,500,000 (2026 estimate)",
    "gdp per capita japan": "approximately $34,000 USD (2026 estimate)",
    "gdp per capita germany": "approximately $52,000 USD (2026 estimate)",
}


def search_tool(query):
    """Simulated search/retrieval tool -- looks up a fact from a small fake KB."""
    key = query.lower().strip()
    return FAKE_KNOWLEDGE_BASE.get(key, f"No result found for '{query}'.")


def calculator_tool(expression):
    """Simulated calculator tool -- evaluates a simple arithmetic expression."""
    try:
        # Only allow basic arithmetic characters for safety in this demo
        allowed = set("0123456789.+-*/() ")
        if not set(expression) <= allowed:
            return "Error: invalid characters in expression."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


TOOLS = {
    "Search": search_tool,
    "Calculator": calculator_tool,
}


# ---- The ReAct trace itself ----
# In a real deployment, an LLM generates each Thought/Action line and the
# calling code executes the Action against the real tool, feeding the
# Observation back into the LLM's next turn. Here, the full trace is shown
# explicitly so every step is inspectable -- this IS what the underlying
# loop looks like when an LLM is actually driving it.

QUESTION = (
    "Which country has the higher total GDP: Japan or Germany? "
    "(Total GDP = population x GDP per capita.)"
)

REACT_TRACE = [
    {
        "thought": (
            "To compare total GDP, I need population and GDP-per-capita for both "
            "countries, then multiply each pair and compare the results. I don't have "
            "these figures memorized reliably enough to trust without checking, so "
            "I'll look them up with the Search tool first."
        ),
        "action": ("Search", "population of japan"),
    },
    {
        "thought": "Got Japan's population. Now I need Japan's GDP per capita.",
        "action": ("Search", "gdp per capita japan"),
    },
    {
        "thought": "Now the same two facts for Germany, starting with population.",
        "action": ("Search", "population of germany"),
    },
    {
        "thought": "And Germany's GDP per capita.",
        "action": ("Search", "gdp per capita germany"),
    },
    {
        "thought": (
            "I now have all four numbers. Total GDP = population x GDP per capita. "
            "I should NOT do this multiplication in my head for large numbers -- "
            "I'll use the Calculator tool to avoid an arithmetic slip, especially "
            "since these are large multi-digit multiplications."
        ),
        "action": ("Calculator", "123000000 * 34000"),
    },
    {
        "thought": "That's Japan's total GDP. Now Germany's, using the same approach.",
        "action": ("Calculator", "83500000 * 52000"),
    },
    {
        "thought": (
            "Japan: ~4.182 trillion. Germany: ~4.342 trillion. Germany's total GDP is "
            "higher despite having a much smaller population, because its GDP per "
            "capita is substantially higher. I now have enough information to answer."
        ),
        "action": None,  # no more tool calls needed -- ready for final answer
    },
]

FINAL_ANSWER = (
    "Germany has the higher total GDP (~$4.34 trillion vs. Japan's ~$4.18 trillion), "
    "even though Japan has a larger population (123M vs 83.5M) -- Germany's "
    "significantly higher GDP per capita (~$52,000 vs ~$34,000) more than compensates."
)


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out("REACT (REASON + ACT) PATTERN -- SIMULATED TOOL-USING SCENARIO")
    out("=" * 100)

    out(f"\nQUESTION: {QUESTION}")
    out("\nAVAILABLE TOOLS: Search (looks up facts), Calculator (evaluates arithmetic)")
    out("\nWHY THIS QUESTION NEEDS ReAct: it requires MULTIPLE pieces of external,")
    out("up-to-date information (population and GDP/capita figures a model shouldn't")
    out("just guess from memory) AND a multi-step calculation that's error-prone to do")
    out("mentally with large numbers (Day 6's CoT experiments showed this exact class")
    out("of error). A single-shot prompt with no tool access would have to either")
    out("hallucinate the figures or refuse to answer confidently.")

    out("\n" + "-" * 100)
    out("THE REACT TRACE")
    out("-" * 100)

    for i, step in enumerate(REACT_TRACE, 1):
        out(f"\n--- Turn {i} ---")
        out(f"Thought: {step['thought']}")
        if step["action"] is not None:
            tool_name, tool_input = step["action"]
            out(f"Action: {tool_name}[{tool_input}]")
            result = TOOLS[tool_name](tool_input)
            out(f"Observation: {result}")
        else:
            out("Action: (none -- sufficient information gathered)")

    out(f"\n--- FINAL ANSWER ---")
    out(FINAL_ANSWER)

    out("\n" + "=" * 100)
    out("THE REACT LOOP, GENERALIZED")
    out("=" * 100)
    out("""
1. THOUGHT: the model reasons, in natural language, about what it knows, what it
   still needs, and what to do next -- this reasoning is made EXPLICIT and visible,
   not hidden inside a single opaque forward pass.
2. ACTION: the model emits a structured call to a specific tool with specific
   arguments (Search[query], Calculator[expression], etc.) -- this is parsed by
   the surrounding orchestration code, not executed by the LLM itself.
3. OBSERVATION: the tool's REAL result is fed back into the model's context as
   the next turn's input -- critically, this is genuine external information the
   model could not have known or reliably computed on its own.
4. REPEAT from Thought, now with the new Observation available as context, until
   the model's Thought concludes it has enough information, at which point it
   gives a Final Answer instead of another Action.

WHY THIS MATTERS: this pattern is what turns a language model into an "agent" --
something that can interact with real systems (databases, APIs, calculators,
web search, code execution) rather than only ever generating text from its own
training-time knowledge. Every "AI agent" product performing multi-step tasks
(browsing, running code, querying APIs) is built on some variant of this
Thought -> Action -> Observation loop.
""")

    with open("outputs/react_pattern_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/react_pattern_results.txt")
