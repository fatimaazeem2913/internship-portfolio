# TF-IDF and Word2Vec Implementation – Day 2 Internship

## Project Overview

This project was completed as part of Day 2 internship tasks. The objective was to move from frequency-based counting representations (TF-IDF) to predictive, distributional representations (Word2Vec) — exposing TF-IDF's ceiling concretely, then learning the same "predict from context" training philosophy that underlies modern LLM pretraining.

The project builds directly on the real corpus scraped in Day 1 (Wikipedia COVID-19 article, Wikipedia Exoplanet article, and a Stack Overflow Q&A thread), and includes a from-scratch TF-IDF implementation verified against `sklearn`, a cosine-similarity retrieval system, a concrete proof of TF-IDF's synonym-blindness, a trained Word2Vec skip-gram model, and a demonstrated polysemy failure case.

---

## Objectives

- Extend Day 1's corpus into a manual TF-IDF implementation, verified against `sklearn`'s `TfidfVectorizer`.
- Build a TF-IDF cosine-similarity document retrieval system.
- Prove, with real sentence pairs, that TF-IDF cannot detect synonymy.
- Train a Word2Vec skip-gram model on the real corpus using `gensim`.
- Test word analogies, nearest-neighbor queries, and re-measure synonym similarity using Word2Vec.
- Demonstrate a genuine polysemy failure case using a real word from the corpus.
- Trace the full conceptual arc: counting → its blindness to meaning → predictive training → its improvement → its remaining ceiling.

---

## Technologies Used

- Python 3
- scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- gensim (`Word2Vec`)
- NLTK (`sent_tokenize`)
- NumPy

---

## Project Structure

```
day-2-tfidf-word2vec
|
|-- README.md
|-- Day2_Complete_Report.docx
|
|-- tfidf_manual.py
|-- tfidf_retrieval.py
|-- synonym_tfidf.py
|-- word2vec_train.py
|-- word2vec_analysis.py
|-- polysemy_demo.py
|
|-- data
|   `-- input_corpus.txt
|
|-- outputs
    |-- tfidf_manual_vs_sklearn.txt
    |-- tfidf_retrieval_results.txt
    |-- synonym_tfidf_results.txt
    |-- word2vec_training_summary.txt
    |-- word2vec_analysis_results.txt
    |-- polysemy_demo_results.txt
    `-- word2vec_day2.model
```

---

## Tasks Performed

### 1. Manual TF-IDF vs. sklearn Verification

TF-IDF was implemented completely from scratch in pure Python on 5 real corpus sentences, replicating sklearn's exact formula (raw term counts, smoothed IDF, L2 normalization), then verified value-by-value against `sklearn.feature_extraction.text.TfidfVectorizer`.

**Output:** `outputs/tfidf_manual_vs_sklearn.txt`

### 2. TF-IDF Cosine-Similarity Retrieval System

A document retrieval system was built over all 1,181 usable sentences in the real corpus, ranking documents against a query by cosine similarity between TF-IDF vectors.

**Output:** `outputs/tfidf_retrieval_results.txt`

### 3. Synonym-Blindness Proof

5 sentence pairs with identical meaning but completely different vocabulary (e.g., "car" vs. "automobile") were tested for TF-IDF similarity.

**Output:** `outputs/synonym_tfidf_results.txt`

### 4. Word2Vec Skip-Gram Training

A skip-gram Word2Vec model was trained on the real corpus using `gensim`, learning dense 100-dimensional vectors for every word that appeared at least twice.

**Output:** `outputs/word2vec_training_summary.txt`, `outputs/word2vec_day2.model`

### 5. Analogies, Nearest Neighbors, and Synonym Re-Measurement

Word analogies and nearest-neighbor queries were tested on real corpus vocabulary, and the Task 3 synonym pairs were re-measured using Word2Vec for direct comparison against the TF-IDF scores.

**Output:** `outputs/word2vec_analysis_results.txt`

### 6. Polysemy Failure Demonstration

A genuinely polysemous word from the real corpus ("light" — used both idiomatically and literally) was tested to show Word2Vec returns an identical vector regardless of context.

**Output:** `outputs/polysemy_demo_results.txt`

---

## Results

The full pipeline was executed successfully against the real Day 1 corpus. The generated outputs demonstrate:

- An exact mathematical match (10 decimal places) between manual and sklearn TF-IDF implementations.
- A working retrieval system correctly ranking real corpus sentences by topical relevance.
- Concrete, zero-similarity proof of TF-IDF's blindness to synonymy across 5 real sentence pairs.
- A trained Word2Vec model with a 2,134-word vocabulary learned from 25,360 real tokens.
- Measurable synonym-detection improvement using Word2Vec over TF-IDF.
- A concrete demonstration of Word2Vec's polysemy failure using a genuinely ambiguous word from the corpus.

---

## Observations

- **TF-IDF's manual implementation matched sklearn to 0.0000000000 difference** — the key detail was that sklearn's default TF is a *raw term count*, not count divided by document length; normalization happens only at the very end via L2 vector scaling.
- **Every single synonym pair scored exactly 0.0000 on TF-IDF**, not just "low" — proving TF-IDF is a purely lexical, string-matching representation with zero concept of meaning.
- **Word2Vec measurably improved synonym detection**: "disease"/"illness" jumped from 0.0000 (TF-IDF) to 0.5031 (Word2Vec), learned with zero manual labels, purely from the predictive skip-gram training objective.
- **Word2Vec's improvement was moderate, not dramatic**, because the training corpus (~25,000 tokens) is small by Word2Vec standards — production embeddings trained on billions of words show far stronger relationships using the same algorithm. This shows embedding quality scales with data volume, not just algorithm choice.
- **Word2Vec completely fails at polysemy**: the word "light," used in two genuinely different senses in the real corpus (idiomatic "shed light on..." vs. literal "reflected light from..."), returned the exact same vector both times, because Word2Vec is a fixed lookup table with no mechanism to consult the current sentence at inference time.
- This directly motivates the need for context-sensitive representations (self-attention in Transformers), which recompute each word's vector dynamically based on its actual surrounding context in each specific sentence.

---

## Challenges Encountered

- The initial `git push` attempt failed because Git had no configured identity on the fresh Linux install (`user.name`/`user.email` were unset) — resolved by configuring Git identity before committing.
- GitHub rejected password-based authentication for `git push` (deprecated for security reasons) — a Personal Access Token was generated as a first attempt, but clipboard copy/paste between the browser and terminal proved unreliable on the Linux Mint setup.
- **SSH key authentication was set up instead** as a more robust long-term solution — an `ed25519` key pair was generated, the public key was added to GitHub's SSH keys, and the repository remote was switched from HTTPS to SSH, resolving the authentication issue permanently for all future pushes.
- A Python virtual environment broke after moving the project folder to a new directory (`~/day-1-nlp-pipeline` → `~/Internship/day-1-nlp-pipeline`), since the venv's internal path references did not update automatically — resolved by deleting and recreating the virtual environment at its new location.

All issues were resolved before the final pipeline run, and the repository was successfully pushed with all outputs intact.

---

## How to Run

Clone the repository:
```
git clone https://github.com/fatimaazeem2913/internship-day2-tfidf-word2vec.git
```

Move into the project directory:
```
cd internship-day2-tfidf-word2vec
```

Set up a virtual environment and install dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn gensim nltk numpy
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Run each script in order (Word2Vec training must run before the analysis/polysemy scripts, since they load its saved model):
```
python3 tfidf_manual.py
python3 tfidf_retrieval.py
python3 synonym_tfidf.py
python3 word2vec_train.py
python3 word2vec_analysis.py
python3 polysemy_demo.py
```

---

## Learning Outcomes

Through this project, the following was learned:

- How to replicate a library's exact mathematical formula from scratch and verify correctness numerically, rather than just trusting library output.
- Why counting-based representations (TF-IDF, Bag-of-Words) are fundamentally blind to synonymy, and why this is a structural limitation, not a tunable parameter.
- The direct conceptual link between Word2Vec's skip-gram training objective ("predict context from a center word") and how GPT-style models are pretrained ("predict the next token from everything before it") — both are self-supervised predictive tasks where the text provides its own training signal.
- Why embedding quality scales with training data volume, using the exact same algorithm — the same scaling principle that motivates training modern LLMs on massive datasets.
- The precise mechanical reason static embeddings fail at polysemy: one vector per word string, frozen after training, with no mechanism to consult context at inference time.
- Why this specific limitation is exactly what motivates self-attention in Transformer architectures.
- How to set up SSH key authentication for Git as a permanent fix for unreliable token/password-based pushes.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 2
