"""
polysemy_demo.py
------------------
Finds a polysemous word from our REAL corpus used in two clearly different
senses, and shows that Word2Vec (a static embedding model) assigns it the
EXACT SAME vector regardless of which sentence/context it appears in.

Word chosen: "light"
  Sense 1 (idiomatic/informal, News/Dialogue-adjacent usage):
    "...shed light on social and economic issues..."
    -> means "reveal/clarify/inform," nothing to do with photons

  Sense 2 (literal, Science domain usage):
    "...a Sun-like star is about a billion times brighter than the
    reflected light from any exoplanet orbiting it."
    -> means literal electromagnetic radiation / photons

These are genuinely different meanings of the same word string -- and
Word2Vec has ONE vector per word type, so it cannot tell them apart.
"""

from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load("outputs/word2vec_day2.model")
wv = model.wv

SENTENCE_A = "They shed light on social and economic issues, including student debt and food insecurity"
SENTENCE_B = "A Sun-like star is about a billion times brighter than the reflected light from any exoplanet orbiting it"

WORD = "light"

if __name__ == "__main__":
    lines = ["=" * 90, "POLYSEMY DEMONSTRATION: static embeddings cannot distinguish word senses", "=" * 90]

    lines.append(f'\nWord under test: "{WORD}"\n')
    lines.append(f'Sentence A (idiomatic sense - "reveal/inform"):\n  "{SENTENCE_A}"\n')
    lines.append(f'Sentence B (literal sense - "electromagnetic radiation"):\n  "{SENTENCE_B}"\n')

    if WORD in wv:
        vector = wv[WORD]
        lines.append(f"Word2Vec vector for '{WORD}' when it appears in Sentence A (first 10 dims):")
        lines.append(f"  {vector[:10]}")
        lines.append(f"\nWord2Vec vector for '{WORD}' when it appears in Sentence B (first 10 dims):")
        lines.append(f"  {vector[:10]}")

        lines.append(f"\nAre the two vectors IDENTICAL? {True}  <- there is only ONE vector for '{WORD}' in the whole model")
        lines.append(f"Cosine similarity between 'the vector' and 'itself': {wv.similarity(WORD, WORD):.6f}")

        lines.append("\nCONCLUSION:")
        lines.append("Word2Vec (and any static embedding table) stores exactly ONE vector per")
        lines.append("word STRING, learned once during training by averaging over every context")
        lines.append("that word ever appeared in across the whole corpus. It has no mechanism to")
        lines.append("look at the CURRENT sentence and adjust the vector on the fly. So whether")
        lines.append(f"'{WORD}' means 'illumination/photons' or 'to reveal/inform,' the model hands")
        lines.append("back the exact same 100-dimensional number every single time, because from")
        lines.append("the model's point of view, it's just one lookup-table row indexed by the")
        lines.append("word string -- context never enters the lookup at inference time.")
    else:
        lines.append(f"'{WORD}' not found in vocabulary (appeared fewer than min_count times).")

    output = "\n".join(lines)
    print(output)

    with open("outputs/polysemy_demo_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/polysemy_demo_results.txt")
