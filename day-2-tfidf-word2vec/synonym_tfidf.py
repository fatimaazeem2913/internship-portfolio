"""
synonym_tfidf.py
-----------------
Takes 5 synonym/paraphrase pairs and measures their TF-IDF cosine similarity.
Since TF-IDF only counts exact word overlap, synonyms sharing NO exact words
will score near zero despite having near-identical meaning.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 5 synonym/paraphrase pairs — same meaning, different words.
# Chosen to be relevant to our Day 1 corpus domains (news/disease, science/space, dialogue/code)
SYNONYM_PAIRS = [
    ("The car broke down on the highway", "The automobile stalled on the road"),
    ("Doctors discovered a new illness", "Physicians identified a novel disease"),
    ("The planet orbits its star", "The world circles its sun"),
    ("The function returns a huge result", "The method yields an enormous output"),
    ("Scientists study the vast universe", "Researchers examine the immense cosmos"),
]

if __name__ == "__main__":
    lines = ["=" * 90, "TF-IDF SIMILARITY ON SYNONYM / PARAPHRASE PAIRS", "=" * 90]
    lines.append("\n(Each pair has the SAME meaning, expressed with DIFFERENT words)\n")

    results = []
    for sent_a, sent_b in SYNONYM_PAIRS:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([sent_a, sent_b])
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        results.append((sent_a, sent_b, sim))

        lines.append(f'A: "{sent_a}"')
        lines.append(f'B: "{sent_b}"')
        lines.append(f"TF-IDF cosine similarity: {sim:.4f}")
        lines.append("")

    avg_sim = sum(r[2] for r in results) / len(results)
    lines.append(f"Average similarity across all 5 pairs: {avg_sim:.4f}")
    lines.append("\nCONCLUSION: despite every pair sharing near-identical MEANING, TF-IDF scores")
    lines.append("are low/near-zero because it only measures exact word overlap after stopword")
    lines.append("removal. 'car' and 'automobile' are treated as completely unrelated tokens —")
    lines.append("TF-IDF has no concept of synonymy or semantic meaning, only string matching.")

    output = "\n".join(lines)
    print(output)

    with open("outputs/synonym_tfidf_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/synonym_tfidf_results.txt")
