"""
tsne_visualization.py
------------------------
Encodes the same ambiguous word ("light") across multiple different-context
sentences using BOTH Word2Vec (static) and a pretrained Transformer (BERT,
contextual), then projects both sets of vectors into 2D with t-SNE to
visually compare: does the word cluster as ONE point (static) or SPREAD OUT
based on meaning (contextual)?

REQUIRES INTERNET ACCESS for the BERT half (run locally). The Word2Vec half
uses the model trained in Day 2 (outputs/word2vec_day2.model).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel

WORD = "light"

# Multiple genuinely different-context sentences using "light" in different senses
CONTEXT_SENTENCES = [
    ("They shed light on the social issues affecting the community.", "idiomatic: reveal/inform"),
    ("The politician tried to shed light on the new policy.", "idiomatic: reveal/inform"),
    ("The reflected light from the exoplanet was extremely faint.", "literal: photons/radiation"),
    ("Sunlight is a natural source of visible light for plants.", "literal: photons/radiation"),
    ("She felt a light breeze as she walked outside.", "literal: physically not heavy"),
    ("He gave a light punishment for the minor mistake.", "figurative: not severe"),
]


def get_word2vec_vectors():
    model = Word2Vec.load("outputs/word2vec_day2.model")
    if WORD not in model.wv:
        raise ValueError(f"'{WORD}' not in Word2Vec vocabulary.")
    # Word2Vec has ONE vector for "light" -- return the SAME vector for every
    # sentence, since that's exactly the point being illustrated.
    vec = model.wv[WORD]
    return [vec.copy() for _ in CONTEXT_SENTENCES]


def get_bert_vectors():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    vectors = []
    for sentence, _ in CONTEXT_SENTENCES:
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state[0]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        positions = [i for i, tok in enumerate(tokens) if tok == WORD]
        if not positions:
            raise ValueError(f"'{WORD}' not found as standalone token in: {tokens}")
        vectors.append(hidden_states[positions[0]].numpy())
    return vectors


def plot_tsne(vectors, labels, title, filename, perplexity=3):
    vectors = np.array(vectors)

    # If every vector is (numerically) identical, t-SNE has zero variance to work
    # with and can crash (div-by-zero -> segfault in some builds). This is itself
    # a meaningful finding worth showing directly rather than working around silently.
    all_identical = np.allclose(vectors, vectors[0], atol=1e-6)
    if all_identical:
        print(f"  NOTE: all {len(vectors)} input vectors are numerically identical --")
        print("  t-SNE cannot compute meaningful distances from zero-variance input.")
        print("  Plotting all points at the same location with tiny visual jitter,")
        print("  labeled clearly as an artifact of the plotting step, not the data.")
        rng = np.random.RandomState(42)
        proj = rng.normal(scale=0.01, size=(len(vectors), 2))  # visual jitter only
        title += "\n(all vectors mathematically identical -- tiny jitter added only so points don't overlap)"
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
        proj = tsne.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    colors = {"idiomatic: reveal/inform": "tab:blue",
              "literal: photons/radiation": "tab:orange",
              "literal: physically not heavy": "tab:green",
              "figurative: not severe": "tab:red"}

    for i, (point, label) in enumerate(zip(proj, labels)):
        plt.scatter(point[0], point[1], c=colors.get(label, "gray"), s=120)
        plt.annotate(f"  sent {i+1}\n  ({label})", (point[0], point[1]), fontsize=8)

    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved plot to {filename}")


if __name__ == "__main__":
    labels = [sense for _, sense in CONTEXT_SENTENCES]

    print("Computing Word2Vec (static) vectors...")
    w2v_vectors = get_word2vec_vectors()

    print("Computing BERT (contextual) vectors (requires internet)...")
    bert_vectors = get_bert_vectors()

    print("\nWord2Vec vectors identical across all sentences (expected):")
    print(f"  All same vector: {all(np.allclose(w2v_vectors[0], v) for v in w2v_vectors)}")

    plot_tsne(w2v_vectors, labels,
              f"Word2Vec (static): '{WORD}' across {len(CONTEXT_SENTENCES)} different contexts",
              "outputs/tsne_word2vec_static.png", perplexity=2)

    plot_tsne(bert_vectors, labels,
              f"BERT (contextual): '{WORD}' across {len(CONTEXT_SENTENCES)} different contexts",
              "outputs/tsne_bert_contextual.png", perplexity=2)

    print("\nCONCLUSION:")
    print("The Word2Vec t-SNE plot collapses to a SINGLE point (all 6 sentences map to the")
    print("exact same vector -- t-SNE may show tiny artificial jitter, but the underlying")
    print("vectors are mathematically identical). The BERT t-SNE plot should show vectors")
    print("SPREAD OUT and GROUPED BY MEANING -- the idiomatic-sense sentences should cluster")
    print("together, separate from the literal-photon-sense sentences, which should cluster")
    print("separately again from the 'not heavy' and 'not severe' senses. This is the direct")
    print("visual proof of context-sensitive vs static representation.")
