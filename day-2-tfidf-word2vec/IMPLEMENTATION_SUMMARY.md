# Day 2 Implementation Summary: TF-IDF to Distributional Embeddings

This document walks through every task from the Day 2 objective, explaining what was built, how it was implemented, and what the results showed.

---

## Task 1: Manual TF-IDF vs. sklearn Verification

**Solution:** `tfidf_manual.py`

**Implementation steps:**
1. Selected 5 real sentences from the Day 1 corpus (mixing News, Science, and Dialogue domains).
2. Tokenized each sentence with a regex pattern matching lowercase letters and apostrophes.
3. Built a sorted vocabulary across all 5 sentences.
4. Computed document frequency (df) for every vocabulary word — how many of the 5 sentences contain it.
5. Computed IDF using sklearn's exact smoothed formula: `ln((1+n)/(1+df)) + 1`.
6. Computed raw term frequency (word count) per sentence.
7. Multiplied TF × IDF for every word/sentence pair, then L2-normalized each resulting vector.
8. Ran the identical 5 sentences through `sklearn.feature_extraction.text.TfidfVectorizer` and compared every value in the resulting matrix against the manual computation.

**Outcome:** Maximum absolute difference between manual and sklearn output: **0.0000000000** — an exact match across all 52 vocabulary terms and all 5 sentences.

**Key learning:** sklearn's default TF-IDF does NOT use the "classic" textbook definition of TF (term count ÷ document length). It uses raw term counts, with document-length normalization happening implicitly via the final L2 normalization step instead. Understanding this distinction was essential to getting an exact match rather than a "close but not quite" result.

---

## Task 2: TF-IDF Cosine Similarity Retrieval System

**Solution:** `tfidf_retrieval.py`

**Implementation steps:**
1. Loaded the full Day 1 corpus and split it into individual sentences using NLTK's `sent_tokenize`.
2. Filtered to sentences between 30–300 characters (removes fragments and overly long run-ons from Wikipedia's dense citation-heavy text).
3. Built a corpus-wide `TfidfVectorizer` (with English stopwords removed) and fit it on all 1,181 usable sentences, producing one TF-IDF vector per sentence.
4. For any incoming query, transformed the query text using the *same fitted vectorizer* (critical — the query must be projected into the same vector space as the documents).
5. Computed cosine similarity between the query vector and every document vector using `sklearn.metrics.pairwise.cosine_similarity`.
6. Sorted documents by similarity score, descending, and returned the top-k matches.

**Outcome:** Tested 3 queries across different domains. The astronomy query ("planets orbiting stars in space") returned highly relevant results with strong similarity scores (top result: 0.4855), since query and document vocabulary overlapped well. The Python/code query also correctly surfaced generator/memory-related sentences.

**Key learning:** cosine similarity measures the *angle* between two vectors, not their magnitude — this matters because a short document and a long document discussing the same topic should still be considered similar, and raw dot-product or Euclidean distance would unfairly penalize the shorter one. Cosine similarity naturally handles this since it's normalized.

---

## Task 3: Synonym-Blindness Proof

**Solution:** `synonym_tfidf.py`

**Implementation steps:**
1. Constructed 5 sentence pairs where each pair expresses the *same meaning* using *completely different vocabulary* (e.g., "car" vs. "automobile," "doctors" vs. "physicians").
2. For each pair independently, fit a fresh 2-document `TfidfVectorizer` and computed cosine similarity between the two resulting vectors.
3. Averaged the similarity across all 5 pairs.

**Outcome:** All 5 pairs scored **exactly 0.0000** similarity. Average: 0.0000.

**Key learning:** this is the clearest possible demonstration that TF-IDF (and Bag-of-Words before it) is a purely *lexical* (string-matching) representation with zero semantic understanding. Two sentences can mean literally the same thing and score as maximally dissimilar, simply because they don't share exact vocabulary. This motivated the need for Task 4's approach.

---

## Task 4: Word2Vec Skip-Gram Training

**Solution:** `word2vec_train.py`

**Implementation steps:**
1. Loaded and cleaned the full Day 1 corpus, split into sentences via NLTK.
2. Tokenized each sentence into lowercase word lists, discarding sentences with fewer than 3 tokens (too short to provide meaningful context windows).
3. Trained a `gensim.models.Word2Vec` model with:
   - `sg=1` (skip-gram mode, rather than CBOW)
   - `vector_size=100` (each word becomes a 100-dimensional vector)
   - `window=5` (considers 5 words on either side of the center word as "context")
   - `min_count=2` (ignores words appearing fewer than twice — too rare to learn a reliable vector)
   - `epochs=50` (extra training passes to compensate for our comparatively small corpus)
4. Saved the trained model for reuse in later tasks.

**Outcome:** Learned a vocabulary of 2,134 unique words from ~25,360 training tokens, each represented as a 100-dimensional dense vector.

**Key learning — the core conceptual bridge of Day 2:** skip-gram's training objective, "predict the surrounding context words given a center word," is a *self-supervised predictive task* — the text supplies its own training signal, no manual labeling required. This is the direct conceptual ancestor of how GPT-style models are pretrained on "predict the next token given everything before it." Different prediction direction, identical underlying philosophy: force a model to get good at predicting language, and it is forced to internally develop representations that capture real semantic and usage patterns as a byproduct.

---

## Task 5: Analogies, Nearest Neighbors, and Synonym Re-Measurement

**Solution:** `word2vec_analysis.py`

**Implementation steps:**
1. Attempted analogy queries using `model.wv.most_similar(positive=[...], negative=[...])` — adapted away from the textbook "king-man+woman=queen" example, since royalty vocabulary doesn't appear in our COVID/exoplanet/Python corpus. Instead tested domain-relevant analogies like `virus + disease − infection`.
2. Ran nearest-neighbor queries (`model.wv.most_similar(word)`) for corpus-relevant words: pandemic, planet, function, virus, star, code.
3. Re-measured the same *concept* of synonym pairs from Task 3, adapted to single words so Word2Vec's word-level vectors could be queried directly (e.g., "disease" vs. "illness" instead of full sentences), and compared results side-by-side against the Task 3 TF-IDF scores.

**Outcome:**
- Analogy `virus + disease − infection` surfaced "coronavirus" in its top-3 results — a genuinely meaningful relationship learned purely from context statistics, without any labeled training data.
- Nearest neighbors for "planet" included "pulsar," "atmosphere," and "dwarf" — real astronomy vocabulary, again learned from context alone.
- Synonym re-test: "disease"/"illness" jumped from **0.0000 (TF-IDF)** to **0.5031 (Word2Vec)** — a clear, measurable improvement. "study"/"research" and "function"/"method" also showed moderate positive similarity (0.37–0.34), versus zero for TF-IDF.

**Key learning:** the improvement is real but modest — because our training corpus (~25k tokens) is tiny by Word2Vec standards, which typically wants millions of tokens to build reliable semantic clusters. This surfaced an important, honest finding: **embedding quality scales with training data volume**, using the exact same algorithm. This is the same scaling principle underlying why modern LLMs are pretrained on datasets orders of magnitude larger than anything feasible in a single afternoon's experiment.

---

## Task 6: Polysemy Failure Demonstration

**Solution:** `polysemy_demo.py`

**Implementation steps:**
1. Searched the real corpus for a word genuinely used in two different senses across different domains.
2. Selected **"light"**: used idiomatically in the News-adjacent text ("shed light on social issues" = reveal/inform) and literally in the Science text ("reflected light from any exoplanet" = electromagnetic radiation/photons).
3. Queried `model.wv["light"]` and printed the resulting vector regardless of which sentence context it was being considered in.
4. Compared the vector returned in both "contexts" to confirm they are identical.

**Outcome:** The vector returned was **bit-for-bit identical** in both cases (cosine similarity of the vector with itself: 1.000000) — trivially true, but this triviality IS the finding.

**Key learning:** Word2Vec (and any static embedding table) stores exactly **one vector per word string**, computed once during training by blending together every context that word ever appeared in. There is no mechanism at lookup time to consult the current sentence and adjust the output — it's a fixed lookup table indexed purely by the word string. This is the fundamental ceiling of *static* (also called "distributional" or "context-independent") embeddings.

---

## Task 7: The Full Report (`REPORT.md`)

Traces the complete arc: TF-IDF's exact mathematical mechanics → its provable blindness to synonymy → Word2Vec's predictive training philosophy (and its direct conceptual link to GPT-style pretraining) → its measurable improvement on synonym detection → its remaining polysemy failure → ending on the open question of what a model would need to do to produce context-dependent vectors.

---

## Overall Key Takeaways From Day 2

1. **Counting ≠ understanding.** TF-IDF is mathematically elegant and exactly reproducible (we proved this to 10 decimal places), but it is fundamentally a string-matching technique with zero concept of meaning.

2. **Prediction is a more powerful training signal than counting.** Word2Vec's skip-gram objective — forcing a model to predict context from a center word — produces representations that capture real semantic relationships (synonyms cluster closer, meaningful analogies emerge) purely as a side effect of getting better at prediction. This is a direct conceptual preview of how GPT-style models are trained at a much larger scale.

3. **Embedding quality scales with data volume**, not just algorithm sophistication. Our modest 25k-token corpus produced real but limited improvements over TF-IDF; production embeddings trained on billions of words show much stronger relationships using the exact same underlying algorithm.

4. **Static embeddings have a hard ceiling: polysemy.** A single word gets exactly one vector no matter what surrounds it, because the lookup happens with no reference to the current sentence. This is a structural limitation, not something that more training data can fix — the architecture itself needs to change.

5. **The open question Day 2 leaves us with — "what would let a model produce a different vector for the same word depending on context?" — is answered directly by the Transformer architecture** studied earlier: self-attention recomputes each word's representation fresh, for every sentence, based on every other word actually present. Day 2's "predict from context" training philosophy plus this missing piece (context-sensitive computation, not just context-informed training) is essentially the recipe for how modern LLMs work.
