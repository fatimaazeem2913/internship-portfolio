# Chain-of-Thought Prompts

These are the two prompt templates used in cot_accuracy_comparison_openai.py.
`{question}` is replaced with each of the 8 reasoning problems at request time.

---

## DIRECT_PROMPT

Answer the following question. Give ONLY the final answer, as briefly as
possible. Do not show any working or explanation.

Question: {question}

Answer:

---

## COT_PROMPT

Answer the following question. Think through the problem step by step,
showing your reasoning explicitly before giving a final answer. Work
through any calculations or logical steps carefully, one at a time.

Question: {question}

Let's think step by step.
