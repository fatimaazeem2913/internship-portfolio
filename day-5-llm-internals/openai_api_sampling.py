"""
openai_api_sampling.py
-------------------------
Demonstrates temperature, top_p, and top-k-style inspection (via
top_logprobs) using the REAL, CURRENT OpenAI Chat Completions API. Provided
for you to run LOCALLY with your own API key -- this sandboxed verification
environment cannot reach api.openai.com (confirmed directly: network
whitelist blocks it).

The sampling ALGORITHMS are identical to the ones implemented from scratch
in sampling_strategies.py -- this file just shows how a production API
exposes the same controls as simple parameters instead of you implementing
the softmax/filtering math by hand.

SETUP (run on your own machine):
    pip install openai
    export OPENAI_API_KEY="sk-...your-key-here..."
    python3 openai_api_sampling.py

NOTE: the Chat Completions API does not expose a direct "top_k" sampling
parameter -- only temperature and top_p control sampling directly. top_k
is demonstrated here via "top_logprobs", which lets you INSPECT the top-k
candidates the model considered at each position (up to k=20), even though
you can't force sampling restricted to only those k through this endpoint.

OpenAI's own docs recommend adjusting EITHER temperature OR top_p, not
both at once, since their combined effect is hard to reason about.
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"   # swap for any chat model you have access to
PROMPT = "The virus"


def generate_with_temperature(prompt, temperature, max_tokens=20):
    """
    temperature: float, 0.0 to 2.0
        0.0   = as close to deterministic as the API allows (always
                near-highest-probability token; OpenAI notes tiny
                floating-point variation can still occur across requests)
        1.0   = the model's default, unmodified distribution
        >1.0  = flatter distribution -- more randomness, more diversity
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def generate_with_top_p(prompt, top_p, max_tokens=20):
    """
    top_p: float in (0, 1]
        Keeps the smallest set of tokens whose cumulative probability
        mass exceeds top_p, then samples only from that set.
        0.1  = only the top 10% of probability mass is eligible (very focused)
        1.0  = no restriction, full distribution eligible
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        top_p=top_p,
        temperature=1.0,   # hold temperature neutral to isolate top_p's effect
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def inspect_top_k_via_logprobs(prompt, k=5):
    """
    There is no top_k sampling parameter in this API. Setting logprobs=True
    plus top_logprobs=k reveals the k highest-probability tokens the model
    considered at the FIRST generated position, letting you inspect what a
    top-k=k restriction would have included -- informational only, does
    not change how the single returned token was actually sampled.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        logprobs=True,
        top_logprobs=k,
    )
    return response.choices[0].logprobs.content[0].top_logprobs


if __name__ == "__main__":
    print("=" * 90)
    print("OPENAI API: TEMPERATURE AND TOP-P SAMPLING (real API, run locally)")
    print("=" * 90)

    print(f"\nModel: {MODEL}")
    print(f"Prompt: \"{PROMPT}\"\n")

    print("--- Temperature sweep ---")
    for temp in [0.0, 0.7, 1.5]:
        text = generate_with_temperature(PROMPT, temperature=temp)
        print(f"\nT={temp}:\n{text}")

    print("\n\n--- Top-p sweep ---")
    for p in [0.1, 0.5, 0.95]:
        text = generate_with_top_p(PROMPT, top_p=p)
        print(f"\ntop_p={p}:\n{text}")

    print("\n\n--- Top-k-style inspection via top_logprobs ---")
    top_candidates = inspect_top_k_via_logprobs(PROMPT, k=5)
    print("Top 5 candidate first tokens and their log-probabilities:")
    for candidate in top_candidates:
        print(f"  {candidate.token!r:<15} logprob={candidate.logprob:.4f}")

    print("\nCompare this to sampling_strategies.py's from-scratch results: the SHAPE")
    print("of the effect (sharper/more deterministic at low temperature or low top_p,")
    print("more diverse at high values) should match exactly, even though the")
    print("underlying model producing the base distribution is vastly more capable")
    print("than our trigram model.")
