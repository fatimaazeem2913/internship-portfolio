"""
stem_lemma_compare.py
----------------------
Removes stopwords, then applies BOTH stemming (PorterStemmer) and
lemmatization (WordNetLemmatizer) to the same tokens, side by side.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in stop_words]


def compare_stem_lemma(tokens):
    rows = []
    for tok in tokens:
        stemmed = stemmer.stem(tok)
        lemmatized = lemmatizer.lemmatize(tok.lower())
        rows.append((tok, stemmed, lemmatized))
    return rows


if __name__ == "__main__":
    with open("outputs/cleaned_corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = word_tokenize(text)
    tokens_no_stop = remove_stopwords(tokens)

    print(f"Original token count: {len(tokens)}")
    print(f"After stopword removal: {len(tokens_no_stop)}")

    # Pick a diverse sample of interesting words to showcase stemming vs lemmatization differences
    sample_words = [
        "running", "runs", "ran", "studies", "studying", "better",
        "discoveries", "orbiting", "researchers", "mice", "cutting",
        "wolves", "flies", "happier", "children", "confirmed"
    ]

    rows = compare_stem_lemma(sample_words)

    print(f"\n{'Word':<15}{'Stemmed':<15}{'Lemmatized':<15}")
    print("-" * 45)
    report_lines = [f"{'Word':<15}{'Stemmed':<15}{'Lemmatized':<15}", "-" * 45]
    for word, stemmed, lemma in rows:
        line = f"{word:<15}{stemmed:<15}{lemma:<15}"
        print(line)
        report_lines.append(line)

    with open("outputs/stem_lemma_comparison.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # also process the real corpus tokens (no stopwords) and save a sample
    real_rows = compare_stem_lemma(tokens_no_stop[:40])
    with open("outputs/stem_lemma_corpus_sample.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Word':<15}{'Stemmed':<15}{'Lemmatized':<15}\n")
        f.write("-" * 45 + "\n")
        for word, stemmed, lemma in real_rows:
            f.write(f"{word:<15}{stemmed:<15}{lemma:<15}\n")

    print("\nResults saved to outputs/stem_lemma_comparison.txt and outputs/stem_lemma_corpus_sample.txt")
