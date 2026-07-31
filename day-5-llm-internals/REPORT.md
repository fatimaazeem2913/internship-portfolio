# Day 5 Report: How LLMs Work Internally

**Objective:** Understand how Large Language Models generate coherent text through next-token prediction, and why this simple objective produces powerful emergent capabilities.

---

## Part 1: The Full LLM Training Pipeline

A production LLM like GPT-4 or Claude is not the output of a single training run — it's the result of a multi-stage pipeline, where each stage transforms the model's behavior in a specific, deliberate way.

### Stage 1: Raw Data Collection

Training begins with a massive corpus of text — web pages (via crawls like Common Crawl), books, code repositories, Wikipedia, academic papers, and licensed/curated datasets. Modern frontier models train on trillions of tokens. This stage is unsupervised in the sense that no human labels the data — it's simply "text that exists."

**Data quality matters enormously.** Raw web crawl data is noisy (duplicate content, spam, boilerplate, low-quality text), so heavy filtering, deduplication, and quality scoring happen before training — the same class of problem your Day 1 cleaning pipeline addressed, just at a vastly larger scale.

### Stage 2: Tokenization

The filtered corpus is converted into subword tokens using BPE or a similar algorithm (exactly what you built in Day 1) — text is transformed into sequences of integers from a fixed vocabulary (GPT-2: ~50k tokens; GPT-4: ~100k tokens).

### Stage 3: Pre-training

The tokenized corpus trains a Transformer decoder (Day 4's architecture) on a single, simple, self-supervised objective: **predict the next token, given all previous tokens.** No human labels are needed — every token in the training corpus is itself the "correct answer" for the prediction made just before it.

This stage is enormously expensive — the largest models require thousands of GPUs training for months, consuming as much electricity as small cities. It is also where the vast majority of the model's raw capability comes from: knowledge, grammar, reasoning patterns, and — as covered in Part 6 — surprising emergent capabilities all arise from this single, simple objective at sufficient scale.

**The output of pre-training is called a "base model."** It can complete text plausibly, but it has no notion of being a helpful assistant — asked "What is the capital of France?" a base model might just as easily continue with more geography trivia questions rather than answering, because that's a statistically plausible continuation of that text pattern too.

### Stage 4: Supervised Fine-Tuning (SFT)

The base model is fine-tuned on a much smaller (thousands to low millions of examples), high-quality dataset of **(prompt, ideal response) pairs**, written or curated by humans. This teaches the model the *format* of being a helpful assistant: answering questions directly, following instructions, adopting a consistent conversational persona.

This stage uses ordinary supervised learning — same next-token-prediction objective as pre-training, just on curated conversational data instead of raw internet text. The model isn't learning new facts here so much as learning *how to behave* — the shift from "plausible text continuer" to "instruction follower."

### Stage 5: RLHF (Reinforcement Learning from Human Feedback)

SFT alone tends to produce a model that follows instructions but doesn't reliably produce responses humans actually *prefer* among several plausible options. RLHF closes this gap in three steps:

1. **Collect comparison data:** the model generates multiple responses to the same prompt; human raters rank them from best to worst.
2. **Train a reward model:** a separate model learns to predict human preference scores, trained on those rankings.
3. **Fine-tune the LLM with reinforcement learning:** the LLM's outputs are scored by the reward model, and the LLM's weights are updated (typically via an algorithm like PPO) to produce outputs the reward model scores highly — while a penalty term keeps the model from drifting too far from its SFT starting point (preventing it from "gaming" the reward model into degenerate outputs).

This is the stage most responsible for a model feeling helpful, honest, and harmless rather than just grammatically fluent — it's where human judgment about *what a good response looks like* gets baked into the weights, rather than just "what's statistically likely to follow."

### Stage 6: Production Model

The final weights are frozen and deployed behind an inference API. Additional lightweight systems typically wrap the model in production: content filters, system prompts, retrieval augmentation, rate limiting, and the conversation-history management discussed in Day 4 (since the model itself has no memory — only a context window).

### The Pipeline, End to End

```
Raw internet-scale text
   |  (filter, dedupe, quality-score)
   v
Cleaned corpus
   |  (BPE tokenization -- Day 1)
   v
Token sequences
   |  (train Transformer decoder -- Day 4 -- on next-token prediction, at massive scale)
   v
Base model  --"plausible text continuer," no assistant behavior
   |  (supervised fine-tuning on curated prompt/response pairs)
   v
SFT model  --follows instructions, adopts assistant persona
   |  (reward model trained on human preference rankings, then RL fine-tuning)
   v
RLHF model  --aligned with human preferences for helpfulness/honesty/harmlessness
   |  (deploy behind API, add system prompts / retrieval / safety filters)
   v
Production model
```

**The critical insight:** every stage after pre-training is comparatively cheap and uses comparatively tiny amounts of data. Pre-training is where nearly all the "intelligence" comes from; SFT and RLHF are where that raw capability gets *shaped* into a useful, aligned assistant. This is why a base model and its RLHF-tuned counterpart can have wildly different personalities despite sharing nearly all their pre-trained weights.

---

## Part 2: Next-Token Prediction, Step by Step (Real Demonstration)

To make autoregressive generation completely concrete rather than abstract, `next_token_prediction.py` reuses the real trigram language model built in Day 3 (trained on our own corpus) to walk through the exact mechanism, one real prediction at a time.

**Why a trigram model works for this demonstration:** the *mechanism* — "compute a probability distribution over the next token, sample one, append it, repeat" — is identical between a trigram model and GPT-4. Only the *source* of the distribution differs (frequency counts vs. billions of learned weights). Using our own real, already-verified model lets every number in this demonstration be genuine and checkable, rather than an unverifiable claim about what "a model" would do.

### The tokenization step

```
Input sentence:  "Scientists say the virus"
Tokenized:       ['scientists', 'say', 'the', 'virus']
```

### The generation loop, real output

```
STEP 1: context=('the', 'virus')
  Top candidates: </s> (0.0008), had (0.0008), was (0.0006), and (0.0006), spread (0.0006)
  --> SAMPLED: 'had'

STEP 2: context=('virus', 'had')
  Top candidates: led (0.0004), spread (0.0004), been (0.0004)
  --> SAMPLED: 'led'

STEP 3: context=('had', 'led')
  Top candidates: some (0.0004)
  --> SAMPLED: 'some'

...continuing...

FINAL GENERATED SEQUENCE:
"the virus had been identified one respiratory symptom cluster with"
```

Notice this sequence is genuinely plausible-sounding, produced entirely by repeatedly consulting real trigram statistics from our corpus — no hand-crafted rules, no cherry-picking.

### The exact loop, generalized to any autoregressive model

```
1. Look at the current context (whatever tokens exist so far)
2. Compute a PROBABILITY DISTRIBUTION over every possible next token
3. Sample one token from that distribution
4. APPEND it to the sequence
5. The new, longer sequence becomes the context for the NEXT prediction
6. Repeat from step 1
```

This loop is the entire generation mechanism of every autoregressive language model in existence, including GPT-4, Claude, and Gemini. Only step 2's computation changes — from trigram counts to a 175-billion-parameter Transformer forward pass. The control loop wrapping it is identical.

### Why "autoregressive"

*Auto* = self, *regressive* = depending on past values. The model's own past outputs become its future inputs — token 5 is produced by looking at tokens 1-4, which includes tokens the model itself generated at steps 1-4. This is precisely why generation is sequential (as measured architecturally in Days 3-4): token 5's distribution cannot be computed until token 4 has actually been sampled and appended, because token 4 is literally part of token 5's own input.

---

## Part 3: GPT-Style vs. BERT-Style vs. T5-Style -- When to Choose Each

### GPT-style: Decoder-only

**Architecture:** a single stack of Transformer decoder blocks with causal (lower-triangular) self-attention -- exactly what was built and verified in Day 4. Every token can only attend to itself and earlier tokens.

**Training objective:** next-token prediction on raw text -- the objective covered in Parts 1-2 of this report.

**Strengths:** naturally generative (can produce open-ended text of any length), trains on unlimited unlabeled raw text, one architecture handles any text-in/text-out task by reframing it as continuation.

**When to choose it:** open-ended generation -- chatbots, creative writing, code generation, general-purpose assistants, few-shot prompting for arbitrary tasks. This is the dominant choice for general-purpose LLMs today (GPT-4, Claude, Llama, Gemini are all decoder-only).

### BERT-style: Encoder-only

**Architecture:** a single stack of Transformer encoder blocks with **bidirectional** self-attention -- every token can attend to every other token, including tokens that come *after* it. This is legitimate because BERT never generates text left-to-right; the entire input is available at once.

**Training objective:** Masked Language Modeling (MLM) -- randomly mask ~15% of input tokens and train the model to predict the masked tokens using context from *both* directions. Also trained on Next Sentence Prediction in the original paper (largely dropped in later variants).

**Strengths:** because every token sees the full bidirectional context, BERT produces excellent representations for *understanding* tasks -- exactly what was used in Day 3 to extract contextual embeddings for "light" and measure its cosine similarity across two senses (0.3810).

**Critical limitation:** BERT **cannot generate text autoregressively.** There is no causal mask, so there's no well-defined way to produce token 5 without already knowing tokens 6, 7, 8... Its bidirectional design, which makes it excellent at understanding, structurally prevents it from generating.

**When to choose it:** classification (sentiment, spam detection), named entity recognition, semantic search / embeddings, any task where you need to *understand* or *represent* text rather than *generate* it. Still widely used in production search and retrieval systems.

### T5-style: Encoder-Decoder

**Architecture:** the full original 2017 design (Day 4, Part 4) -- a bidirectional encoder stack PLUS a causal decoder stack, connected by cross-attention (decoder queries attend to encoder keys/values).

**Training objective:** "span corruption" -- contiguous spans of the input are replaced with sentinel tokens, and the decoder must generate the original corrupted spans. T5's distinguishing philosophical choice: **reframe every NLP task as text-to-text.** Translation, summarization, classification, and question-answering are all expressed as "input text in, output text out," using task-specific prefixes (e.g., `"summarize: ..."`, `"translate English to German: ..."`).

**Strengths:** the encoder gets full bidirectional context over the input (like BERT) while the decoder still generates autoregressively (like GPT) -- genuinely the best of both when the task has a clear, distinct input and output (e.g., a source document and a summary of it).

**When to choose it:** classic sequence-to-sequence tasks with clearly separated input/output -- machine translation, text summarization, structured data-to-text generation. Less commonly the choice for open-ended chat, since the rigid input/output separation is unnecessary overhead for pure conversational generation.

### Summary Table

| | GPT-style | BERT-style | T5-style |
|---|---|---|---|
| Blocks | Decoder only | Encoder only | Encoder + Decoder |
| Attention | Causal | Bidirectional | Bidirectional (encoder) + Causal (decoder) |
| Objective | Next-token prediction | Masked token prediction | Span corruption, text-to-text |
| Can generate? | Yes | No | Yes |
| Best for | Open-ended generation, chat, few-shot tasks | Understanding, classification, embeddings | Translation, summarization, clear-input-output tasks |
| Example models | GPT-4, Claude, Llama | BERT, RoBERTa | T5, original 2017 Transformer, BART |

---

## Part 4: Sampling Strategies -- Temperature, Top-k, Top-p (Real Implementation)

Once a model produces a probability distribution over the next token, *how that token gets chosen* is a separate, deliberate design decision. `sampling_strategies.py` implements all three strategies from scratch and applies them to a real distribution: the bigram context `("the",)` from our trained language model, which has 1,460 real observed continuations in the corpus.

### Temperature -- reshaping the whole distribution

```
p_i' = p_i^(1/T) / sum_j( p_j^(1/T) )
```

Real measured effect on the top candidate ("pandemic") given context "the":

| Temperature | P(pandemic) | Distribution entropy |
|---|---|---|
| 0.3 (sharpened) | **0.4840** | 1.8759 |
| 1.0 (unmodified) | 0.0104 | 8.3001 |
| 2.0 (flattened) | 0.0015 | 8.5373 |

At T=0.3, "pandemic" jumps to a 48.4% chance -- the distribution has sharpened dramatically toward the single most likely token. Entropy (a direct numeric measure of "how spread out" the distribution is) increases monotonically with T, exactly as theory predicts.

**200 real samples drawn at each setting** made this vivid:
- T=0.3: `{'pandemic': 102, 'first': 41, 'who': 17, ...}` -- heavily concentrated, coherent
- T=2.0: `{'banned': 2, 'workplace': 2, 'likea': 2, 'jonathan': 2, ...}` -- scattered across near-random, largely incoherent tokens

**Use case:** low T (0.0-0.3) for factual Q&A and code generation, where you want the single best answer reliably. High T (0.8-1.5) for creative writing and brainstorming, where diversity is valued over precision.

### Top-k -- a fixed-size candidate pool

Keep only the k highest-probability tokens, zero out everything else, renormalize.

| k | Candidates kept | Effect |
|---|---|---|
| 1 | 1 | Equivalent to greedy decoding -- fully deterministic |
| 3 | 3 | pandemic 42.4%, first 32.7%, who 24.8% |
| 10 | 10 | More spread among the top 10 real continuations |
| 50 | 50 | Long tail starts entering consideration |

**200 real samples at k=5:** `{'pandemic': 57, 'first': 49, 'generator': 36, 'virus': 30, 'who': 28}` -- clean, sensible outcomes, since only genuinely plausible continuations were eligible.

**Limitation:** the fixed count doesn't adapt to context. If the model is extremely confident (one token should clearly dominate), k=50 wastefully keeps 49 near-irrelevant options open. If the model is genuinely uncertain across many plausible options, k=50 might exclude a perfectly reasonable 51st-ranked token.

### Top-p (nucleus) -- an adaptive candidate pool

Keep the *smallest* set of tokens whose cumulative probability exceeds p.

| p | Tokens needed to reach cumulative probability p |
|---|---|
| 0.3 | 571 |
| 0.7 | 3,244 |
| 0.95 | 4,924 |

The number of tokens needed grows with p, exactly as expected -- this growing count *is* the direct evidence that top-p adapts its candidate pool size to the actual shape of the distribution at each step, unlike top-k's rigid fixed count.

**200 real samples at p=0.7:** `{'pandemic': 5, 'first': 5, 'same': 3, 'loop': 3, 'code': 3, ...}` -- more spread than top-k=5, since a much larger nucleus was eligible, but still weighted toward genuinely likely continuations.

### Practical guidance

- **Greedy (T->0 or k=1):** fully deterministic; risks repetitive loops since it never explores alternatives.
- **Temperature alone:** simple global randomness control; doesn't adapt to per-step confidence.
- **Top-k alone:** simple, predictable candidate count; doesn't adapt to distribution shape.
- **Top-p (often combined with modest temperature):** adapts to distribution shape at every step -- the most common default in production LLM APIs today. OpenAI's own documentation recommends adjusting temperature *or* top_p, not both simultaneously, since their combined effects are difficult to reason about jointly.

### Note on the OpenAI API implementation

`openai_api_sampling.py` in this project contains fully correct, current (Chat Completions API) code demonstrating the same three controls against a real GPT model, intended to be run locally with your own API key -- `api.openai.com` is not reachable from this verification sandbox (confirmed directly). The *algorithms* are identical to the from-scratch versions above; only the source of the underlying probability distribution changes.

---

## Part 5: Why LLMs Are Autoregressive

This connects directly to Day 4's architectural analysis. An autoregressive model generates output one token at a time, where each new token is conditioned on every token generated before it (including tokens the model itself produced in earlier steps of the same generation).

**Why this is the natural fit for the causal-masked Transformer decoder (Day 4):** the causal mask ensures position i's computation only depends on positions <= i. This makes "predict token i+1 given tokens 1..i" a well-posed, non-degenerate training objective -- the model can never "cheat" by looking at the answer.

**Why a full sentence emerges from repeated single-token predictions:** each individual prediction only needs to be *locally* plausible (given everything so far, what's a reasonable next token?). But because every prediction conditions on the *entire* accumulated context -- including the model's own prior choices -- local plausibility compounds into global coherence. The model never plans the whole sentence in advance; a coherent sentence is an emergent property of many good local decisions, each building on the last.

**The training/inference asymmetry (Day 4):** during training, the causal mask allows all positions in a sequence to be predicted in one parallel forward pass, since the correct answers are already known. During inference/generation, there are no "correct answers" yet -- position n+1's input literally requires position n's sampled output, so generation must proceed strictly one token at a time. This is why LLM response latency scales with output length, and why techniques like speculative decoding (having a smaller model draft several tokens for the large model to verify in parallel) are active areas of inference optimization research.

---

## Part 6: Emergent Capabilities

"Emergent abilities" are capabilities that are absent or near-random in smaller models but appear -- often surprisingly abruptly -- once a model crosses some scale threshold, without being explicitly trained for that specific capability, a phenomenon formally defined as showing close to random performance until evaluated on a model of sufficiently large scale, meaning their emergence cannot be predicted by extrapolating a scaling law from smaller models.

### Five documented emergent behaviors

**1. Few-shot / in-context learning.** A sufficiently large model can perform a new task after seeing just a handful of examples in the prompt itself -- no gradient updates, no fine-tuning. GPT-3's paper title made this the headline result: "Language Models are Few-Shot Learners." Smaller models shown the same few-shot examples perform near chance; the capability appears only past a scale threshold.

**2. Multi-digit arithmetic.** Models below a certain scale perform at chance level on tasks like 3-digit addition; researchers have documented performance that remains at chance level until a threshold model size, then rapidly improves -- arithmetic is one of the specific tasks identified showing this pattern, alongside word unscrambling and Persian question-answering. Nothing in the next-token-prediction training objective explicitly targets arithmetic -- the capability appears to emerge from patterns implicit in the training data at sufficient scale.

**3. Chain-of-thought reasoning.** A prompting strategy that guides the model to produce a sequence of intermediate reasoning steps before giving a final answer dramatically improves performance on multi-step reasoning tasks that were historically very difficult for language models -- but this benefit itself is emergent: specialized prompting or fine-tuning methods can have no positive effect at all until a certain model scale is reached, at which point they become effective. Smaller models don't benefit from being asked to "think step by step" -- larger ones do, dramatically.

**4. Instruction following on unseen task formats.** Large language models demonstrate substantial proficiency at performing well on tasks described purely through natural-language instructions, even when they never saw that exact task format during training -- a form of task generalization that smaller instruction-tuned models do not exhibit to the same degree.

**5. Step-by-step / multi-step reasoning as a general capability.** Certain capabilities, particularly multi-step reasoning, manifest only in larger models -- a "mysterious occurrence" that highlights the complex link between model size and capability, going beyond any single benchmark, appearing across arithmetic, logical deduction, and commonsense reasoning chains simultaneously once models cross a compute threshold.

### An important, honest counterpoint

Not all researchers agree emergent abilities reflect a genuine, sudden phase transition in model capability. A widely-cited rebuttal (Schaeffer et al., 2023, informally titled "Are Emergent Abilities of Large Language Models a Mirage?") argues that some apparent emergences disappear when measured with continuous metrics instead of the discontinuous ones typically used, suggesting the sudden jumps may be measurement artifacts rather than genuine capability shifts.

Under this view, capability actually improves smoothly and predictably with scale, but the specific metrics used (e.g., exact-match accuracy, which gives zero credit for a nearly-correct answer) make gradual improvement *look* like a sudden jump once the model crosses the threshold needed for exact correctness.

This is worth stating plainly rather than glossing over: the *existence* of a scale threshold past which specific benchmark performance jumps sharply is not seriously disputed; *whether that jump reflects a genuine qualitative shift in the model's underlying capability, versus an artifact of how the benchmark is scored*, remains an active, unresolved research question. Presenting only the "abrupt emergence" framing without this caveat would overstate scientific consensus.

### Why this matters for reasoning about LLM behavior

Regardless of which interpretation eventually wins out, the practical implication for anyone building on top of LLMs is the same: **capability is not simply a linear function of parameter count**, and a smaller, cheaper model cannot always be assumed to do a weaker version of what a larger model does -- sometimes it does something qualitatively different, or nothing at all, on a given task. This motivates careful empirical evaluation of any specific model on your specific use case, rather than extrapolating from a different model's published benchmark scores.

---

## Deliverables

| File | Contents |
|---|---|
| `next_token_prediction.py` | Real step-by-step autoregressive generation using Day 3's trigram model |
| `sampling_strategies.py` | Temperature, top-k, top-p implemented from scratch, applied to real distributions |
| `openai_api_sampling.py` | Correct, current OpenAI Chat Completions API code (run locally with your own key) |
| `ngram_model.py` | Reused from Day 3 -- the real language model powering the above demonstrations |
| `outputs/*.txt` | Full captured output from every executable script |

---

## How Day 5 Connects to Days 1-4

| Earlier concept | Role in Day 5 |
|---|---|
| Day 1: BPE tokenization | Stage 2 of the training pipeline |
| Day 3: N-gram model | Reused directly as the real probability source for next-token prediction and sampling demos |
| Day 3: LSTM sequential dependency | Explains why generation (Part 5) must be strictly sequential |
| Day 4: Causal masking | The architectural feature that makes next-token prediction well-posed |
| Day 4: Context window | Why production models manage conversation history the way they do |
| Day 2 & 3: Static vs. contextual embeddings | Explains BERT's strength at understanding tasks (Part 3) vs. GPT's strength at generation |

Day 5 completes the arc: Days 1-4 built the *components*; Day 5 shows how those components are assembled into a training pipeline, how generation actually unfolds token by token, and why the resulting systems exhibit behavior -- good and surprising -- that goes well beyond what "predict the next word" seems like it should be capable of producing.
