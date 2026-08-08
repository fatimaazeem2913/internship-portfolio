"""
persona_switcher.py
-----------------------
Implements a persona-switching system: 3 AI assistant personas (formal,
casual, technical), toggled purely via the system role, answering the
SAME question -- so any difference in output is attributable entirely to
the system prompt, isolating the persona's effect from the question's.

SETUP:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python3 persona_switcher.py
"""

import os
from google import genai
from google.genai import types
from prompt_loader import load_prompts

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

PERSONAS = ["PERSONA_FORMAL", "PERSONA_CASUAL", "PERSONA_TECHNICAL"]


def ask_as_persona(persona_key, question, temperature=0.5):
    """
    Toggles persona purely by swapping the system_instruction -- the
    user's question and every other parameter stays identical, so the
    persona system prompt is the ONLY variable being changed.
    """
    prompts = load_prompts("prompts/personas.md")
    system_prompt = prompts[persona_key]

    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    return response.text.strip()


class PersonaSession:
    """
    A simple stateful wrapper letting an application toggle personas
    mid-conversation -- e.g. a support bot that switches from casual to
    technical mode when a user asks an advanced question.
    """
    def __init__(self, default_persona="PERSONA_CASUAL"):
        self.current_persona = default_persona

    def switch_to(self, persona_key):
        assert persona_key in PERSONAS, f"Unknown persona: {persona_key}"
        self.current_persona = persona_key

    def ask(self, question, temperature=0.5):
        return ask_as_persona(self.current_persona, question, temperature)


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out("PERSONA-SWITCHING SYSTEM: 3 PERSONAS, SAME QUESTION")
    out("=" * 90)

    prompts = load_prompts("prompts/personas.md")
    question = prompts["PERSONA_TEST_QUESTION"]
    out(f"\nTest question (identical across all 3 personas): \"{question}\"\n")

    for persona_key in PERSONAS:
        out("-" * 90)
        out(f"PERSONA: {persona_key}")
        out("-" * 90)
        system_prompt = prompts[persona_key]
        out(f"System prompt: {system_prompt}\n")
        answer = ask_as_persona(persona_key, question)
        out(f"Answer:\n{answer}\n")

    out("\n" + "=" * 90)
    out("DEMONSTRATING RUNTIME PERSONA SWITCHING (a single session, toggled mid-use)")
    out("=" * 90)
    session = PersonaSession(default_persona="PERSONA_CASUAL")
    out(f"\nSession starts as: {session.current_persona}")
    out(f"Q: {question}")
    out(f"A: {session.ask(question)}\n")

    session.switch_to("PERSONA_TECHNICAL")
    out(f"Session switched to: {session.current_persona}")
    out(f"Q: {question}")
    out(f"A: {session.ask(question)}\n")

    out("=" * 90)
    out("WHAT TO VERIFY IN REAL OUTPUT")
    out("=" * 90)
    out("""
The three answers should differ in STYLE, not in the underlying factual
correctness of the advice given -- all three personas should identify
genuinely valid causes of slow loading (unoptimized images, no caching,
server response time, unminified assets, etc.). What should change:
  - FORMAL: complete sentences, no contractions, measured tone
  - CASUAL: contractions, relaxed phrasing, approachable
  - TECHNICAL: dense terminology, no basic-concept explanations, terse

This is the same principle as Day 6's Role/Persona anatomy component,
demonstrated as working, swappable, runtime-toggleable code rather than
a single static prompt.
""")

    with open("outputs/persona_switcher_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/persona_switcher_results.txt")
