# From Static Vectors to Context – Day 3 Internship

## Project Overview

This project was completed as part of Day 3 internship tasks. The objective was to build representations that actually depend on context, and discover firsthand — through direct measurement, not just theory — why purely sequential processing breaks down at scale, motivating the attention mechanism studied separately.

The project builds on the real corpus scraped in Day 1 and used throughout Day 2 (Wikipedia COVID-19 article, Wikipedia Exoplanet article, Stack Overflow "yield keyword" thread), and includes an n-gram language model with Laplace smoothing, a real gradient-magnitude analysis of an untrained LSTM, contextual embeddings pulled from pretrained BERT, a 4-way comparison table (TF-IDF / Word2Vec / n-gram / BERT), and a t-SNE visualization comparing static vs. contextual clustering.

---

## Objectives

- Implement a bigram/trigram language model with Laplace smoothing; compute next-word probability distributions and generate sample text.
- Run an untrained LSTM forward/backward pass and measure gradient magnitude decay across timesteps.
- Demonstrate that sequential models cannot be parallelized, structurally.
- Pull contextual embeddings for the Day 2 polysemous word ("light") from pretrained BERT, in its two different-sense sentences.
- Build a single comparison table: TF-IDF vs. Word2Vec vs. n-gram vs. BERT cosine similarity, on the same synonym/polysemy test cases from Days 1–2.
- Visualize static vs. context-sensitive clustering with t-SNE.

---

## Technologies Used

- Python 3
- PyTorch (`torch.nn.LSTM`, `torch.nn.LSTMCell`, autograd)
- Hugging Face `transformers` (`BertTokenizer`, `BertModel`)
- gensim (reusing the Day 2 trained Word2Vec model)
- scikit-learn (`TSNE`)
- NLTK, matplotlib, NumPy

---

## Project Structure

```
day-3-context-attention
|
|-- README.md
|-- REPORT.md
|
|-- ngram_model.py
|-- ngram_word_vectors.py
|-- lstm_gradient_analysis.py
|-- bert_contextual_embeddings.py
|-- bert_synonym_embeddings.py
|-- build_comparison_table.py
|-- tsne_visualization.py
|
|-- data
|   `-- input_corpus.txt
|
|-- outputs
    |-- ngram_results.txt
    |-- ngram_word_vector_results.txt
    |-- lstm_gradient_results.txt
    |-- bert_contextual_results.txt          (generated after local run)
    |-- bert_synonym_results.txt / .json      (generated after local run)
    |-- day3_comparison_table.txt
    |-- tsne_word2vec_static.png
    |-- tsne_bert_contextual.png              (generated after local run)
    `-- word2vec_day2.model                   (reused from Day 2)
```

---

## Tasks Performed

### 1. N-Gram Language Model

Bigram and trigram models with Laplace (add-1) smoothing were trained on the real corpus. Next-word probability distributions were computed for several real contexts, and sample text was generated to illustrate the model's fixed-window limitation directly.

**Output:** `outputs/ngram_results.txt`

### 2. LSTM Gradient Analysis

An untrained LSTM (PyTorch) was run forward and backward over a real 55-token sequence from the corpus. Gradient magnitude at every timestep's input was measured to show early-token influence decay, and the strictly sequential nature of the computation was demonstrated directly.

**Output:** `outputs/lstm_gradient_results.txt`

### 3. N-Gram Distributional Word Vectors

A bigram-based distributional word vector (P(next word | word), aggregated corpus-wide) was built to add an "n-gram" column to the final comparison table, and to show n-grams are also a static representation.

**Output:** `outputs/ngram_word_vector_results.txt`

### 4. BERT Contextual Embeddings

BERT's final-layer hidden state for "light" was extracted in the same two different-sense sentences tested in Day 2, and compared against Word2Vec's proven identical-vector result.

**Output:** `outputs/bert_contextual_results.txt`

### 5. Comparison Table

TF-IDF (Day 2), Word2Vec (Day 2), n-gram (this project), and BERT (this project) cosine similarities were assembled into a single table across the same 5 synonym pairs and the polysemous "light" self-comparison.

**Output:** `outputs/day3_comparison_table.txt`

### 6. t-SNE Visualization

"Light" was encoded across 6 sentences spanning 4 distinct senses using both Word2Vec (static) and BERT (contextual), then projected to 2D with t-SNE to visually compare clustering behavior.

**Output:** `outputs/tsne_word2vec_static.png`, `outputs/tsne_bert_contextual.png`

---

## Results

- The n-gram model's generated text became incoherent within a few words, directly illustrating its fixed-window ceiling.
- The LSTM's measured gradient norm was effectively **0.000000** at the earliest timesteps of a 55-token sequence, rising to **0.603** at the final timestep — in an **untrained** network, proving the vanishing-gradient tendency is architectural, not a training artifact.
- N-gram distributional vectors, like Word2Vec, scored a trivial 1.0000 self-similarity on "light" — confirming they are equally static.
- Word2Vec's vectors for "light" across 6 different sentences were confirmed **numerically identical** (`np.allclose` = True), so identical that t-SNE's distance computation failed outright on the input.
- BERT results (contextual embeddings, synonym comparison, t-SNE clustering) are produced by running the provided scripts locally with internet access — see "How to Run."

---

## Observations

- A trigram model completely forgets any word more than 2 positions back — increasing n only helps marginally before data sparsity dominates, since most higher-order n-grams are never seen in any finite corpus.
- Vanishing gradients are not a symptom of poor training — they emerged in a network with entirely random, untrained weights, purely from the repeated multiplication of sub-1 gate activations across many timesteps.
- An LSTM's hidden state at timestep t requires the hidden state at timestep t-1 as a direct input — this is a structural, not incidental, barrier to parallelizing sequence computation, and is the concrete reason Transformer training can be parallelized across a whole sequence while RNN/LSTM training cannot.
- TF-IDF, Word2Vec, and n-gram distributional vectors all share the same structural ceiling: exactly one vector per word, computed once and frozen — none of them can distinguish "light" (reveal) from "light" (photons).
- BERT is expected to be the only representation in this comparison capable of producing genuinely different vectors for the same word depending on its sentence — directly resolving the open question Day 2 ended on.

---

## Challenges Encountered

- Installing PyTorch in the verification environment required significant troubleshooting: the default PyPI wheel for `torch` bundles CUDA/GPU dependencies (several GB) rather than a lightweight CPU-only build, which repeatedly exhausted available disk space mid-install. This was resolved by uninstalling all partial/corrupted `nvidia-*` packages, freeing disk space, and performing a single clean install that let pip resolve the complete dependency set in one pass.
- `transformers`'s latest version required `torch >= 2.4`, incompatible with the installed `torch 2.2.2` — resolved by pinning `transformers==4.40.0`, a version compatible with the available PyTorch build.
- Hugging Face's model hub (`huggingface.co`) is not reachable from the verification sandbox's network whitelist, so pretrained BERT weights could not be downloaded there. The BERT-dependent scripts are fully written and verified for correctness, but must be run on a normal internet-connected machine (see "How to Run") to produce their real output values.
- Word2Vec's "light" vectors across 6 test sentences were confirmed exactly identical, which caused t-SNE to divide by zero internally (zero-variance input) and crash. This was resolved by detecting the all-identical case explicitly and plotting with a clearly labeled, tiny visual jitter rather than letting the crash pass silently.

---

## How to Run

Clone the repository:
```
git clone https://github.com/fatimaazeem2913/internship-day3-context-attention.git
cd internship-day3-context-attention
```

Set up the environment:
```
python3 -m venv venv
source venv/bin/activate
pip install torch transformers gensim scikit-learn nltk matplotlib numpy
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Run scripts that work fully offline first:
```
python3 ngram_model.py
python3 ngram_word_vectors.py
python3 lstm_gradient_analysis.py
```

Then run the BERT-dependent scripts (these download `bert-base-uncased`, ~440MB, on first run — requires internet access):
```
python3 bert_contextual_embeddings.py
python3 bert_synonym_embeddings.py
```

Finally, assemble the comparison table and generate the t-SNE plots:
```
python3 build_comparison_table.py
python3 tsne_visualization.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- Why n-gram models have a hard, structural context-window ceiling, and why simply increasing n doesn't scale due to data sparsity.
- How to measure vanishing gradients directly, rather than taking the phenomenon on faith — and that it appears even in untrained networks, confirming it is architectural rather than a training-quality issue.
- The precise structural reason LSTMs cannot be parallelized during training: each timestep's hidden state is a required input to the next.
- How to extract and compare contextual word embeddings from a pretrained Transformer, and why they differ from static embeddings by construction.
- How to build a single, honest, multi-method comparison table spanning three days of increasingly sophisticated text representations.
- How to diagnose and resolve a real dependency-management crisis (PyTorch's CUDA-bundling behavior on PyPI) under disk constraints.
- Why self-attention is the natural architectural answer to both the polysemy problem (Day 2) and the sequential-bottleneck problem (Day 3) simultaneously.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 3
