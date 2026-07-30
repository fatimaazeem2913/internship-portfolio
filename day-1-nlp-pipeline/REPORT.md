# Day 1 Report: Classic NLP Pipeline & Tokenization

**Corpus:** `data/input_corpus.txt` — 3 domains (News, Science, Dialogue), scraped/simulated with HTML + Markdown noise deliberately included to test the cleaning pipeline.

---

## 1. Text Cleaning

Raw text contained HTML tags (`<h1>`, `<p>`, `<b>`, `<i>`), Markdown syntax (`##`, `**bold**`), numbers, and stray punctuation. Regex-based cleaning (`clean_text.py`) stripped all of it.

**Before:**
```
### DOMAIN: NEWS ###
The globalCOVID-19 pandemic(also known as thecoronavirus pandemic), caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), began with an outbreak inWuhan, China, in December 2019...

**After:**
```
DOMAIN NEWS The globalCOVID pandemic also known as thecoronavirus pandemic , caused by severe acute respiratory syndrome coronavirus SARS CoV , began with an outbreak inWuhan, China, in December . It spread to other parts of Asia andthen worldwidein early . TheWorld Health Organization WHO declared...

**Observation:** Numbers like `2026` and `5%` were stripped entirely (by design, per task spec), which occasionally leaves an awkward gap ("rates by ." / "more than in early trading"). In a real pipeline you'd decide whether to *remove* numbers or *replace* them with a placeholder token like `<NUM>` to preserve sentence structure — worth flagging as a design tradeoff.

**New observation from real data:** stripping HTML tags via regex sometimes merges adjacent words together (`globalCOVID`, `inWuhan`) when the original tag boundary was the only thing separating them. A more robust cleaner would replace tags with a space rather than an empty string before removing them.

---

## 2. Stemming vs. Lemmatization

Ran `PorterStemmer` and `WordNetLemmatizer` on the same word list side by side (`stem_lemma_compare.py`):(Run on a real scraped corpus of 27,230 tokens — 17,435 remained after stopword removal.)

| Word | Stemmed (Porter) | Lemmatized (WordNet) |
|---|---|---|
| running | run | running |
| runs | run | run |
| ran | ran | ran |
| studies | studi | study |
| studying | studi | studying |
| discoveries | discoveri | discovery |
| orbiting | orbit | orbiting |
| researchers | research | researcher |
| mice | mice | **mouse** |
| cutting | cut | cutting |
| wolves | wolv | **wolf** |
| flies | fli | **fly** |
| children | children | **child** |
| confirmed | confirm | confirmed |

### Where each succeeds / fails

**Stemming (Porter):**
- ✅ Fast, rule-based, no dictionary lookup needed — good for search/indexing at scale.
- ❌ Produces non-words: `studi`, `discoveri`, `wolv`, `fli` — not real English words, unreadable if shown to a user.
- ❌ Completely fails on irregular plurals: `mice` stays `mice`, `wolves` becomes the garbled `wolv` instead of `wolf`.
- ❌ Inconsistent: `ran` stays `ran` (doesn't reduce to `run`) because Porter only strips suffixes, it doesn't know verb conjugation.

**Lemmatization (WordNet):**
- ✅ Produces real dictionary words: `study`, `discovery`, `wolf`, `fly`, `child`, `mouse`.
- ✅ Correctly handles irregular forms **when given the right POS**: `mice → mouse`, `children → child`.
- ❌ **Defaults to noun mode** if no POS tag is passed — this is the biggest gotcha. `running`, `orbiting`, `cutting`, `confirmed` were NOT reduced to their base verb form (`run`, `orbit`, `cut`, `confirm`) because WordNetLemmatizer assumed they were nouns. You must explicitly pass `pos='v'` for verbs to get correct results.
- ❌ Slower than stemming (dictionary lookup) and needs a POS tagger for best accuracy — more moving parts.

**Takeaway:** stemming is a blunt, fast axe; lemmatization is a precise but pickier scalpel that needs POS context to actually shine.

---

## 3. Tokenization: Character vs. Word vs. Subword (BPE / WordPiece)

Trained a BPE tokenizer and a WordPiece tokenizer from scratch on our small corpus (`tokenization_compare.py`), vocab size capped at 300 to intentionally force subword splitting on unseen/rare words.

**Sentence:** `"Researchers discovered an unbelievably habitable exoplanet."`

| Method | Token count | Tokens |
|---|---|---|
| Character-level | 59 | `R, e, s, e, a, r, c, h, e, r, s, ...` |
| Word-level | 7 | `Researchers, discovered, an, unbelievably, habitable, exoplanet, .` |
| BPE (subword) | 24 | `Res, e, arch, ers, d, is, co, ve, re, d, an, u, n, be, l, i, e, v, ab, ly, habitable, ex, oplanet, .` |
| WordPiece (subword) | 21 | `R, ##esearch, ##ers, d, ##isco, ##ver, ##ed, an, u, ##n, ##b, ##el, ##i, ##e, ##v, ##ab, ##ly, habitable, ex, ##oplanet, .` |

### Where each succeeds / fails

**Character-level:**
- ✅ Zero out-of-vocabulary (OOV) problem — literally any string can be represented (just ~100 possible characters).
- ✅ Tiny vocabulary size.
- ❌ Sequences become extremely long (59 tokens for one short sentence) — expensive for a transformer, since attention cost grows quadratically with sequence length.
- ❌ Loses word-level meaning entirely; the model has to relearn "what makes a word" from scratch, which needs much more data/capacity.

**Word-level:**
- ✅ Short, intuitive sequences (7 tokens) — matches human intuition of "words."
- ✅ Preserves whole-word meaning directly.
- ❌ Massive OOV problem: any word not seen during training (typos, new slang, rare technical terms, other languages) becomes a single `[UNK]` token, and the model loses all information about it.
- ❌ Vocabulary size explodes for large corpora (hundreds of thousands of unique words), making the embedding/output layers huge.

**Subword (BPE / WordPiece):**
- ✅ Best of both worlds: common words stay as single tokens (`habitable` stayed whole in both — it was frequent/simple enough in our corpus), rare/unseen words get broken into meaningful sub-pieces instead of becoming `[UNK]`.
- ✅ Fixed, manageable vocabulary size regardless of corpus size.
- ❌ Splits can look arbitrary on words absent from training data: `unbelievably` → `u, n, be, l, i, e, v, ab, ly` is a messy split because our tiny training corpus (a few paragraphs, vocab size 300) never saw this word or its parts often enough to merge them into sensible chunks. With a production-scale vocab (e.g., 30k–50k, as in real BERT/GPT tokenizers) this would look much cleaner, e.g. `un`, `believ`, `ably`.
- ❌ BPE vs WordPiece differ subtly in *how* they choose merges (BPE = frequency of adjacent pairs; WordPiece = likelihood/score-based), which is why their splits aren't identical even trained on the same data (see `##` continuation markers WordPiece uses vs BPE's plain concatenation).

**Practical note:** the messy small-corpus splits above are expected — real models (GPT, BERT, Claude) train subword tokenizers on billions of words, so common English morphemes ("un-", "-ably", "-tion") get proper standalone tokens instead of fragmenting to near-character level like we see here.

---

## 4. POS Tagging (spaCy)

Ran `en_core_web_sm` on representative sentences (`pos_tagging.py`):

**Sentence:** `"Astronomers identified a rocky exoplanet orbiting within the habitable zone."`

| Token | POS | Fine Tag | Meaning |
|---|---|---|---|
| Astronomers | NOUN | NNS | plural noun |
| identified | VERB | VBD | past-tense verb |
| rocky | ADJ | JJ | adjective |
| exoplanet | NOUN | NN | singular noun |
| orbiting | VERB | VBG | gerund/present participle |
| habitable | ADJ | JJ | adjective |
| zone | NOUN | NN | singular noun |

Bucketed by category:
- **Nouns:** Astronomers, exoplanet, zone
- **Verbs:** identified, orbiting
- **Adjectives:** rocky, habitable

This is exactly the kind of tagging that should feed into a smarter lemmatizer (see Section 2's `pos='v'` gotcha) — `orbiting` tagged as `VBG` (verb) tells the lemmatizer to reduce it to `orbit`, not leave it untouched.

---

## 5. Bag-of-Words (from scratch)

Built a pure-Python BoW extractor (`bow_extractor.py`, no sklearn) on 4 sentences spanning all 3 domains:

- **Vocabulary:** 27 unique words across the 4 sentences.
- Example dict for `"Jordan fixed the tokenization bug but the lemmatizer still misbehaves."`:
  ```python
  {'jordan': 1, 'fixed': 1, 'the': 2, 'tokenization': 1, 'bug': 1,
   'but': 1, 'lemmatizer': 1, 'still': 1, 'misbehaves': 1}
  ```
  Note `'the': 2` — BoW captures raw frequency, which is why it's called a "bag" (order is thrown away, only counts matter).

**Where BoW succeeds / fails:**
- ✅ Extremely simple, interpretable, cheap to compute — great baseline for classic ML classifiers (Naive Bayes, logistic regression on text).
- ❌ Throws away word order entirely — "dog bites man" and "man bites dog" produce identical BoW vectors.
- ❌ No notion of meaning/similarity — "happy" and "joyful" are just as unrelated as "happy" and "car" in this representation.
- ❌ Sparse and huge at scale — a real corpus vocabulary could have 50k+ dimensions, mostly zeros per sentence.

This is precisely the gap that word embeddings (Word2Vec, GloVe) and later transformer-based contextual embeddings were built to close.

---

## Summary Table

| Technique | Best for | Key weakness |
|---|---|---|
| Regex cleaning | Removing structural noise (HTML/MD/numbers) | Can accidentally break sentence flow if too aggressive |
| Stemming | Fast, crude normalization at scale | Produces non-words, fails irregular forms |
| Lemmatization | Accurate, readable base forms | Needs correct POS to work well; slower |
| Character tokenization | Zero OOV, tiny vocab | Very long sequences |
| Word tokenization | Intuitive, short sequences | Large OOV problem |
| Subword (BPE/WordPiece) | Balance of vocab size + OOV handling | Messy on tiny/undertrained vocabularies |
| POS tagging | Grammatical structure, feeds lemmatization | Language/domain-specific, needs a trained model |
| Bag-of-Words | Simple baseline features | No order, no semantics |

**Overall lesson from Day 1:** every one of these classic techniques is a stepping stone toward the same goal — turning messy human text into structured numeric input a model can learn from — and each one trades off *simplicity* for *loss of information* somewhere. Transformers (which you're studying alongside this) essentially solve the weaknesses of BoW and word-level tokenization simultaneously: subword tokenization avoids the OOV problem, and self-attention captures the word-order/context information that BoW throws away.
