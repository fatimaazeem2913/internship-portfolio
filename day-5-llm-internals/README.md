# How LLMs Work Internally – Day 5 Internship

## Project Overview

This project was completed as part of Day 5 internship tasks. The objective was to understand how Large Language Models generate coherent text through next-token prediction, and why this simple objective produces powerful, sometimes surprising, emergent capabilities.

The work covers the complete LLM training pipeline (raw data through RLHF to a production model), a real step-by-step demonstration of autoregressive generation using the trigram language model built in Day 3, a from-scratch implementation of temperature/top-k/top-p sampling verified against real probability distributions, a structured comparison of GPT/BERT/T5-style architectures, and researched documentation of five emergent capabilities with an honest discussion of the ongoing scientific debate around them.

---

## Objectives

- Document the full LLM training pipeline: raw data -> tokenization -> pre-training -> SFT -> RLHF -> production model.
- Demonstrate next-token prediction step by step on a tokenized sentence.
- Compare GPT-style (decoder-only), BERT-style (encoder-only), and T5-style (encoder-decoder) architectures, and when to choose each.
- Implement temperature, top-p, and top-k sampling strategies and show how each changes generated output.
- Explain why LLMs are autoregressive and how a full sentence emerges from repeated single-token predictions.
- Research and document 5 emergent capabilities not explicitly trained for.

---

## Technologies Used

- Python 3
- NumPy
- The n-gram language model built in Day 3 (reused directly as a real, verifiable source of next-token probability distributions)
- OpenAI Python SDK (reference implementation only -- requires the user's own API key and local internet access)

---

## Project Structure

```
day-5-llm-internals
|
|-- README.md
|-- REPORT.md
|
|-- ngram_model.py                  (reused from Day 3)
|-- next_token_prediction.py
|-- sampling_strategies.py
|-- openai_api_sampling.py
|
|-- data
|   `-- input_corpus.txt            (same real corpus from Days 1-3)
|
|-- outputs
    |-- next_token_prediction_results.txt
    `-- sampling_strategies_results.txt
```

---

## Tasks Performed

### 1. LLM Training Pipeline Documentation

Each of the six stages -- raw data collection, tokenization, pre-training, supervised fine-tuning, RLHF, and production deployment -- is documented in `REPORT.md` Part 1, explaining what happens at each stage and why it exists.

### 2. Next-Token Prediction, Step by Step

`next_token_prediction.py` tokenizes a real input sentence and walks through autoregressive generation one prediction at a time, using the actual trigram model trained on our Day 1/2 corpus -- including a real fallback to bigram statistics when a trigram context has no training data, demonstrating the data-sparsity problem directly.

**Output:** `outputs/next_token_prediction_results.txt`

### 3. Architecture Comparison

GPT-style, BERT-style, and T5-style architectures are compared in `REPORT.md` Part 3 across attention pattern, training objective, generation capability, and best-fit use cases.

### 4. Sampling Strategies From Scratch

`sampling_strategies.py` implements temperature scaling, top-k filtering, and top-p (nucleus) filtering entirely from scratch in NumPy, applied to a real probability distribution (1,460 real observed continuations from our bigram model), with 200 real samples drawn under each configuration to make the effect on actual output concrete.

**Output:** `outputs/sampling_strategies_results.txt`

### 5. OpenAI API Reference Implementation

`openai_api_sampling.py` contains correct, current (Chat Completions API) code demonstrating the same three sampling controls against a real GPT model. Provided for local execution with your own API key, since this verification environment cannot reach `api.openai.com`.

### 6. Emergent Capabilities Research

Five documented emergent behaviors -- few-shot/in-context learning, multi-digit arithmetic, chain-of-thought reasoning, instruction following on unseen formats, and general multi-step reasoning -- are researched and cited in `REPORT.md` Part 6, alongside an honest discussion of the Schaeffer et al. (2023) counterargument that some apparent emergence may be a measurement artifact rather than a genuine capability phase transition.

---

## Results

- **Real autoregressive generation produced genuinely plausible text**: "the virus had been identified one respiratory symptom cluster with," emerging purely from real trigram statistics with no hand-crafted rules.
- **Temperature's effect measured concretely**: P(pandemic) went from 0.0104 (T=1.0) to 0.4840 (T=0.3) to 0.0015 (T=2.0), with entropy increasing monotonically as predicted by theory.
- **200 real samples at each setting made the qualitative difference vivid**: T=0.3 produced heavily concentrated, coherent word choices; T=2.0 produced scattered, largely incoherent tokens.
- **Top-p's adaptivity proven numerically**: the number of tokens needed to reach cumulative probability p grew from 571 (p=0.3) to 3,244 (p=0.7) to 4,924 (p=0.95) -- direct evidence that top-p's candidate pool size adapts to the distribution's shape, unlike top-k's fixed count.
- **Five emergent capabilities documented with citations**, alongside the important caveat that whether these represent genuine phase transitions or measurement artifacts remains an active research question.

---

## Observations

- The generation loop -- compute a distribution, sample, append, repeat -- is architecturally identical between a simple trigram model and a frontier LLM. Only the mechanism computing the distribution changes in complexity; the control loop wrapping it does not.
- A trigram context frequently has zero training observations (data sparsity, discussed first in Day 3), requiring a backoff to a lower-order model -- a real, concrete illustration of why purely count-based language models hit a hard ceiling regardless of how they're combined.
- Temperature's effect is far more dramatic on a richly-observed bigram context (1,460 real counts) than on a sparse trigram context, where heavy Laplace smoothing flattens the distribution almost uniformly regardless of temperature -- an important practical lesson about how smoothing and sampling interact.
- Top-p's adaptive candidate count is not just a theoretical advantage -- it was directly measurable: the nucleus size changed by nearly an order of magnitude across three p values on the same underlying distribution.
- The "emergent abilities" literature contains a genuine, unresolved scientific disagreement (sudden phase transition vs. measurement artifact of discontinuous metrics) that is easy to gloss over in a superficial treatment, but understanding both sides is necessary for reasoning carefully about what scale actually buys you.

---

## Challenges Encountered

- `api.openai.com` is not reachable from this verification environment's network whitelist (confirmed directly via a blocked request), so `openai_api_sampling.py` could not be executed here. It was written and cross-checked against current OpenAI API documentation (Chat Completions endpoint, `top_logprobs` parameter) rather than the older, now-secondary Completions endpoint, and is intended for local execution with your own API key.
- The first version of the sampling demonstration used a sparse trigram context, which produced a nearly-uniform distribution dominated by Laplace smoothing (5,259-word vocabulary against very few real observed counts) -- this made temperature and top-p's effects numerically uninteresting (e.g., top-p needing 1,500+ tokens even at p=0.3). Switching to a bigram context with 1,460 real observed continuations produced a far more realistic, sharply-peaked distribution where every sampling strategy's effect became clearly visible in the numbers.
- The first seed sentence chosen for the next-token prediction demo ("The virus causes a serious") produced a trigram context never seen in training, causing generation to halt after one step. This was fixed by verifying candidate seed contexts against the model's actual training data first, and by adding a genuine bigram backoff mechanism rather than simply picking a seed by trial and error.

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-5-llm-internals
```

Install dependencies:
```
pip install numpy nltk
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Run the fully offline scripts:
```
python3 next_token_prediction.py
python3 sampling_strategies.py
```

For the OpenAI API reference implementation (requires your own API key and internet access):
```
pip install openai
export OPENAI_API_KEY="sk-...your-key-here..."
python3 openai_api_sampling.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- The complete pipeline turning a raw text corpus into a deployed, aligned assistant, and specifically why pre-training (cheap per-example, unlabeled, enormous in scale) and RLHF (expensive per-example, human-labeled, comparatively tiny) play such different roles in shaping final model behavior.
- That autoregressive generation is a genuinely simple loop -- distribution, sample, append, repeat -- verified directly by reusing an already-built, fully-understood language model rather than treating a production LLM as an unexaminable black box.
- The precise mathematical and practical differences between temperature, top-k, and top-p sampling, including why top-p's adaptive candidate size is a real, measurable advantage over top-k's fixed count.
- A structured, decision-oriented understanding of when to choose GPT-style, BERT-style, or T5-style architectures, grounded in the attention-pattern and training-objective differences studied in Day 4.
- Why LLM generation is inherently sequential at inference time even though training is fully parallel -- the same training/inference asymmetry examined architecturally in Day 4.
- That "emergent capabilities" is a genuinely contested area of active research, not a settled fact -- and why presenting only one side of that debate would be scientifically dishonest.
- How to build a working, verifiable demonstration of a concept (sampling strategies) using an already-available, self-built model when the originally-specified tool (OpenAI API) is unreachable in the current environment, while still providing fully correct reference code for the originally-specified approach.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 5
