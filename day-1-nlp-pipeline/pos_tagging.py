"""
pos_tagging.py
--------------
Extracts Part-of-Speech tags using spaCy, and buckets tokens into
nouns, verbs, adjectives, and other grammatical categories.
"""

import spacy

nlp = spacy.load("en_core_web_sm")

SAMPLE_SENTENCES = [
    "Global markets surged as investors grew more confident about rate cuts.",
    "Astronomers identified a rocky exoplanet orbiting within the habitable zone.",
    "Jordan fixed the tokenization bug but the lemmatizer still misbehaves.",
]


def tag_sentence(sentence):
    doc = nlp(sentence)
    return [(tok.text, tok.pos_, tok.tag_, spacy.explain(tok.pos_)) for tok in doc]


def bucket_by_category(sentence):
    doc = nlp(sentence)
    buckets = {"NOUN": [], "VERB": [], "ADJ": [], "ADV": [], "OTHER": []}
    for tok in doc:
        if tok.pos_ in buckets:
            buckets[tok.pos_].append(tok.text)
        elif tok.pos_ == "PROPN":
            buckets["NOUN"].append(tok.text)  # treat proper nouns as nouns bucket
        else:
            buckets["OTHER"].append((tok.text, tok.pos_))
    return buckets


if __name__ == "__main__":
    lines = ["=" * 90, "POS TAGGING RESULTS (spaCy)", "=" * 90]

    for sent in SAMPLE_SENTENCES:
        lines.append(f"\nSENTENCE: {sent}")
        lines.append(f"{'Token':<15}{'POS':<8}{'Tag':<8}{'Explanation'}")
        lines.append("-" * 70)
        for text, pos, tag, explain in tag_sentence(sent):
            lines.append(f"{text:<15}{pos:<8}{tag:<8}{explain}")

        buckets = bucket_by_category(sent)
        lines.append(f"\n  Nouns: {buckets['NOUN']}")
        lines.append(f"  Verbs: {buckets['VERB']}")
        lines.append(f"  Adjectives: {buckets['ADJ']}")
        lines.append(f"  Adverbs: {buckets['ADV']}")
        lines.append(f"  Other: {buckets['OTHER']}")

    output = "\n".join(lines)
    print(output)

    with open("outputs/pos_tagging_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/pos_tagging_results.txt")
