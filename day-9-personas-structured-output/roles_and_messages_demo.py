"""
roles_and_messages_demo.py
------------------------------
Demonstrates the three-role message structure: system (persistent
instructions), user (human input), assistant (model history). Uses
Gemini as the primary, free API (Day 8's established pattern) -- the
role-separation CONCEPT is identical across every major provider, only
the exact parameter names differ slightly.

WHY THREE ROLES, NOT ONE BIG PROMPT:
Mixing instructions and conversation into a single blob of text forces
the model to guess which parts are "rules to always follow" vs. "things
the user just said." Separating them into distinct roles lets:
  - the SYSTEM role persist unchanged across an entire conversation,
    acting as a constant behavioral contract
  - the USER role carry only the human's actual input
  - the ASSISTANT role carry the model's OWN prior replies, so multi-turn
    context works correctly (the model can see what IT said before,
    distinct from what the human said)

SETUP:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python3 roles_and_messages_demo.py
"""

import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "You are a customer support assistant for a software company. "
    "Always be concise (2-3 sentences max). Never make up information "
    "about pricing -- if asked about pricing and it wasn't provided in "
    "context, say you'll connect them with sales."
)


def build_conversation_history():
    """
    Builds a multi-turn conversation as a list of role-tagged turns.
    Gemini represents this as a list of types.Content objects, each
    tagged with role="user" or role="model" (Gemini's name for the
    assistant role) -- the SYSTEM role is passed separately via
    system_instruction in the config, not as part of this list, since
    it's meant to persist outside the turn-by-turn exchange.
    """
    return [
        types.Content(role="user", parts=[types.Part(text="Hi, does your app work offline?")]),
        types.Content(role="model", parts=[types.Part(text="Yes, our app has full offline mode -- changes sync automatically once you're back online.")]),
        types.Content(role="user", parts=[types.Part(text="Great. What does the Pro plan cost?")]),
    ]


def run_multiturn_conversation():
    """
    Sends the full conversation history in one request. The model's reply
    to the SECOND user message will be informed by the FIRST exchange
    (it knows the app has offline mode), AND by the system instruction
    (it should decline to guess at pricing, per the system rule above) --
    demonstrating that all three roles are actually being used together.
    """
    history = build_conversation_history()
    response = client.models.generate_content(
        model=MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    return response


if __name__ == "__main__":
    print("=" * 90)
    print("SYSTEM / USER / ASSISTANT ROLE SEPARATION DEMO")
    print("=" * 90)

    print(f"\n[SYSTEM ROLE -- persists across the whole conversation]\n{SYSTEM_INSTRUCTION}")

    print("\n[CONVERSATION HISTORY -- user and assistant/model turns]")
    for turn in build_conversation_history():
        role_label = "USER" if turn.role == "user" else "ASSISTANT"
        print(f"  {role_label}: {turn.parts[0].text}")

    print("\n[SENDING FULL 3-ROLE PAYLOAD TO THE MODEL...]")
    response = run_multiturn_conversation()

    print(f"\n[MODEL'S NEW REPLY]\n{response.text}")

    print("\n" + "=" * 90)
    print("WHAT TO VERIFY IN THE REAL OUTPUT")
    print("=" * 90)
    print("""
1. Does the reply correctly decline to state a specific Pro plan PRICE,
   per the system instruction ("never make up pricing")? This proves the
   SYSTEM role's rule persisted and was actually applied, not just present.
2. Does the reply stay concise (2-3 sentences), per the same system rule?
3. Does the reply make sense as a CONTINUATION of the offline-mode
   exchange, rather than treating "What does the Pro plan cost?" as an
   isolated, context-free question? This proves the ASSISTANT role's
   prior turn was genuinely used as context, not just the latest USER
   message in isolation.
""")
