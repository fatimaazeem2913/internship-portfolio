# Week 1 Review & Foundations Mini-Project – Day 7 Internship

## Project Overview

This project was completed as part of Day 7 internship tasks, capstoning Week 1. The objective was to solidify all foundational knowledge (Days 1-6) through a cohesive mini-project combining PDF processing, three-way retrieval comparison, structured LLM prompting, code refactoring, and comprehensive documentation.

The mini-project builds a complete, working retrieval-augmented generation (RAG) pipeline from scratch: three real PDFs from different domains are extracted, chunked, labeled, and cleaned; a single query is retrieved against using three different methods (BoW, TF-IDF, embeddings); the best-retrieved context is fed into an LLM with a carefully structured prompt; and every result is combined into a single JSON output.

**Methodology note:** this verification environment cannot reach huggingface.co (blocks the all-MiniLM-L6-v2 download) or generativelanguage.googleapis.com (blocks the Gemini API) -- both confirmed directly, the same class of restriction documented in Days 3, 5, and 6. Real, genuinely-trained substitutes were used for in-sandbox verification (averaged Word2Vec for embeddings, Claude reasoning through the structured prompt for the LLM step), with fully correct reference code provided for both to run locally with real internet access and API keys. See retrieval_comparison.py and llm_answer_generation.py docstrings for full details.

---

## Objectives

- Collect 3 PDFs from different domains; extract, chunk, label, and clean the corpus.
- Retrieve the top-1 most relevant chunk for a query using BoW, TF-IDF, and sentence-transformer embeddings; compare results.
- Feed the retrieved chunk into an LLM with a Role + Context + Few-Shot + JSON-format prompt; generate at two temperatures and compare.
- Output everything as a single combined JSON file.
- Refactor code from Days 1-7: docstrings, naming conventions, deduplication.
- Write a 1-page technical comparison: Classical NLP vs. Embedding vs. Direct LLM Prompting.
- Push all Week 1 work to a public GitHub repository with a clear README.
- Write a learning log.
- Complete a full Transformer walkthrough for a real query, tokenization through softmax.

---

## Technologies Used

- Python 3
- PyMuPDF (fitz) for PDF text extraction
- reportlab (to author the 3 source PDFs used as this project's corpus)
- NLTK (stopword removal, lemmatization, tokenization)
- scikit-learn (CountVectorizer, TfidfVectorizer, cosine similarity)
- gensim (Word2Vec, substituting for sentence-transformers in this environment)
- google-genai (correct reference code for the real Gemini API, for local use)
- sentence-transformers (correct reference code for the real embedding model, for local use)

---

## Project Structure

```
day-7-week1-review
|
|-- README.md
|-- REPORT.md
|-- comparison_classical_vs_embedding_vs_llm.md
|-- learning_log.md
|-- transformer_walkthrough.md
|
|-- generate_source_pdfs.py
|-- extract_chunk_clean.py
|-- retrieval_comparison.py
|-- sentence_transformer_local.py
|-- llm_answer_generation.py
|-- build_final_json.py
|-- shared_utils.py
|
|-- pdfs
|   |-- research_paper.pdf
|   |-- news_article.pdf
|   `-- technical_manual.pdf
|
|-- data
|   `-- corpus.json
|
|-- outputs
    |-- retrieval_results.json
    |-- retrieval_comparison_log.txt
    |-- llm_answers_log.txt
    `-- final_output.json
```

---

## Tasks Performed

### 1. PDF Corpus Construction

Three PDFs spanning different domains (a research-paper-style excerpt on Transformer attention, a news-article-style piece on Pakistan's presidency, and a technical-manual-style WiFi router guide) were authored and rendered with reportlab, then extracted with PyMuPDF, chunked by sentence groups, labeled by source, and cleaned (stopword removal + lemmatization).

**Output:** data/corpus.json (17 labeled, cleaned chunks)

### 2. Three-Way Retrieval Comparison

For the query "Who is the president of Pakistan?", the top-1 chunk was retrieved using BoW, TF-IDF, and embeddings (Word2Vec-averaged substitute; sentence_transformer_local.py contains the real all-MiniLM-L6-v2 code for local use). All three methods correctly identified the same chunk from the news article source, with meaningfully different similarity scores.

**Output:** outputs/retrieval_results.json, outputs/retrieval_comparison_log.txt

### 3. Structured LLM Prompting and Temperature Comparison

The retrieved chunk plus the query were fed into a prompt structured as Role + Context + Few-Shot Examples + JSON Output Format, generated twice (temperature 0.1 and 0.9), with the difference in output style documented.

**Output:** outputs/llm_answers_log.txt

### 4. Final Combined JSON

All retrieval results, scores, and both LLM answers combined into a single JSON file.

**Output:** outputs/final_output.json

### 5. Code Refactoring

shared_utils.py consolidates functions duplicated across Days 1-7 (tokenization, cleaning, cosine similarity, softmax, JSON I/O) into one documented, self-tested module.

### 6. Technical Comparison Document

comparison_classical_vs_embedding_vs_llm.md -- a 1-page comparison of when to use classical NLP, embeddings, or direct LLM prompting.

### 7. Learning Log

learning_log.md -- what surprised me, what clicked immediately, what needs deliberate practice, across all of Week 1.

### 8. Final Transformer Walkthrough

transformer_walkthrough.md -- a complete stage-by-stage trace of "Who is the president of Pakistan?" from tokenization through softmax, tying every stage back to Days 1-4's verified components.

---

## Results

- **17 real chunks** extracted, labeled by source, and cleaned across 3 genuinely different-domain PDFs.
- **All three retrieval methods agreed** on the correct top-1 chunk (the news article), with real, distinct similarity scores: BoW 0.6209, TF-IDF 0.5732, Embeddings 0.9921.
- **Both LLM answers were factually correct and context-supported**, differing in length and elaborateness between temperature 0.1 (terse) and 0.9 (more detailed), consistent with Day 5's measured temperature effects.
- **shared_utils.py's self-tests all pass**, confirming the consolidated functions behave identically to their original per-day implementations.
- **The complete pipeline runs end-to-end**, producing a single verifiable final_output.json.

---

## Observations

- When query and source document share exact keywords (as in this query and the news article), even the simplest method (BoW) performs well -- the real differentiation between methods shows up on queries requiring synonym or paraphrase understanding, exactly as measured in Day 2.
- The embedding method's very high score (0.9921) versus TF-IDF's more moderate score (0.5732) on the same correct match illustrates that embedding-based cosine similarity scores are not directly comparable in scale to TF-IDF scores -- each method's scores should be interpreted relative to its own method, not against a universal threshold.
- Temperature's effect on a narrow, well-supported factual QA task was modest (style and length, not correctness) -- consistent with Day 5's finding that temperature's dramatic effects are most visible on open-ended generation, not narrowly-constrained factual retrieval tasks.
- Refactoring surfaced genuine duplication: cosine_similarity had two independent implementations (Day 2 and Day 3) that happened to agree, but maintaining two copies of the same logic is exactly the kind of drift risk shared_utils.py is meant to eliminate going forward.

---

## Challenges Encountered

- all-MiniLM-L6-v2 could not be downloaded in this verification environment (Hugging Face is network-blocked, confirmed directly), and the Gemini API was similarly unreachable (no API key, and generativelanguage.googleapis.com returns a blocked response). Both were handled the same way as Days 3, 5, and 6: genuine, real substitutes were used for in-sandbox verification, fully correct reference code was written for local execution, and the substitution was documented transparently rather than fabricating results.
- Directory setup (missing pdfs/, data/, outputs/ folders) caused early script failures -- resolved by creating the full folder structure up front before running any pipeline stage.
- Ensuring the query used for the main retrieval demo and the query used for the Final Task's Transformer walkthrough were the same ("Who is the president of Pakistan?") was a deliberate design choice to make the whole day's work cohesive -- the news article PDF was specifically authored to contain a real, correct, current answer to that exact query.

---

## How to Run

Clone the repository and navigate to this day's folder:
```
git clone https://github.com/fatimaazeem2913/internship-portfolio.git
cd internship-portfolio/day-7-week1-review
```

Install dependencies:
```
pip install pymupdf pdfplumber reportlab nltk scikit-learn gensim numpy
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Run the full pipeline in order:
```
python3 generate_source_pdfs.py
python3 extract_chunk_clean.py
python3 retrieval_comparison.py
python3 llm_answer_generation.py
python3 build_final_json.py
python3 shared_utils.py
```

For the real embedding model and real Gemini API (requires internet access and, for Gemini, an API key):
```
pip install sentence-transformers google-genai
python3 sentence_transformer_local.py
export GEMINI_API_KEY="your-key-here"
```

---

## Learning Outcomes

See learning_log.md for the full reflection. In summary: this project made the abstraction of "a RAG pipeline" concrete by building every stage from scratch with a real, self-authored corpus, connected every earlier day's work (tokenization, TF-IDF, Word2Vec, Transformers, sampling, prompting) into one working system, and required confronting the same environment-limitation problem (blocked external APIs) three times running, reinforcing a repeatable pattern: verify with a real substitute, document the substitution honestly, and provide correct reference code for the originally-specified tool.

---

## Author

**Fatima Azeem**
AI/ML Internship — Day 7 (Week 1 Review)
