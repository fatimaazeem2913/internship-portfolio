"""
build_comparison_table.py
-----------------------------
Assembles the final Day 3 comparison table: TF-IDF vs Word2Vec (static) vs
n-gram (static) vs Transformer/BERT (contextual) cosine similarity, all on
the same synonym pairs (from Day 2) and the same polysemous word ("light",
from Day 2's polysemy_demo.py).

Run this LAST, after bert_contextual_embeddings.py and bert_synonym_embeddings.py
have been run locally (so their output files exist).
"""

import json
import os

# --- Real, verified values from Day 2 (TF-IDF and Word2Vec) ---
TFIDF_SYNONYM_SCORES = {
    "virus/pathogen": 0.0000,
    "disease/illness": 0.0000,
    "planet/world": 0.0000,
    "study/research": 0.0000,
    "function/method": 0.0000,
}
WORD2VEC_SYNONYM_SCORES = {
    "virus/pathogen": None,  # not in Day 2's small vocabulary
    "disease/illness": 0.5031,
    "planet/world": 0.1009,
    "study/research": 0.3756,
    "function/method": 0.3404,
}
TFIDF_LIGHT_SELF_SIM = 1.0000   # TF-IDF is also a static, per-corpus representation
WORD2VEC_LIGHT_SELF_SIM = 1.000000  # proven identical in Day 2

# --- Real values from THIS script's own n-gram run ---
NGRAM_SYNONYM_SCORES = {
    "virus/pathogen": None,   # 'pathogen' not in this corpus's vocabulary either
    "disease/illness": 0.0000,
    "planet/world": 0.1997,
    "study/research": 0.0000,
    "function/method": 0.5193,
}
NGRAM_LIGHT_SELF_SIM = 1.000000


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else "N/A"


def load_bert_synonym_scores():
    path = "outputs/bert_synonym_results.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_bert_light_similarity():
    path = "outputs/bert_contextual_results.txt"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read()
    for line in content.splitlines():
        if "Cosine similarity between the two contextual vectors" in line:
            return float(line.split(":")[-1].strip())
    return None


if __name__ == "__main__":
    bert_synonyms = load_bert_synonym_scores()
    bert_light = load_bert_light_similarity()

    lines = ["=" * 100]
    lines.append("DAY 3 COMPARISON TABLE: TF-IDF vs Word2Vec vs N-gram vs Transformer (BERT)")
    lines.append("=" * 100)

    if bert_synonyms is None or bert_light is None:
        lines.append("\n*** NOTE: BERT results not found. Run bert_synonym_embeddings.py and")
        lines.append("*** bert_contextual_embeddings.py locally FIRST (they need internet access")
        lines.append("*** to download bert-base-uncased), then re-run this script.\n")

    header = f"{'Pair':<20}{'TF-IDF':<12}{'Word2Vec':<12}{'N-gram':<12}{'BERT (contextual)':<18}"
    lines.append("\n" + header)
    lines.append("-" * len(header))

    pairs = ["virus/pathogen", "disease/illness", "planet/world", "study/research", "function/method"]
    for pair in pairs:
        tfidf_v = fmt(TFIDF_SYNONYM_SCORES[pair])
        w2v_v = fmt(WORD2VEC_SYNONYM_SCORES[pair])
        ngram_v = fmt(NGRAM_SYNONYM_SCORES[pair])
        bert_v = fmt(bert_synonyms.get(pair)) if bert_synonyms else "(run locally)"
        lines.append(f"{pair:<20}{tfidf_v:<12}{w2v_v:<12}{ngram_v:<12}{bert_v:<18}")

    lines.append("")
    lines.append(f"{'light (self, 2 senses)':<20}{fmt(TFIDF_LIGHT_SELF_SIM):<12}{fmt(WORD2VEC_LIGHT_SELF_SIM):<12}{fmt(NGRAM_LIGHT_SELF_SIM):<12}{fmt(bert_light) if bert_light else '(run locally)':<18}")

    lines.append("\n" + "=" * 100)
    lines.append("READING THE TABLE:")
    lines.append("- TF-IDF, Word2Vec, and N-gram are all STATIC representations: one fixed")
    lines.append("  vector per word, so 'light' compared with itself is trivially 1.0000 no")
    lines.append("  matter which of its two senses is being discussed -- none of these three")
    lines.append("  methods can tell the senses apart.")
    lines.append("- BERT is the only CONTEXTUAL representation here: it should show light's")
    lines.append("  self-similarity meaningfully BELOW 1.0 (genuinely different vectors for")
    lines.append("  the two senses), while still showing improved synonym detection over")
    lines.append("  TF-IDF for pairs like disease/illness -- resolving Day 2's open question.")

    output = "\n".join(lines)
    print(output)

    with open("outputs/day3_comparison_table.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/day3_comparison_table.txt")
