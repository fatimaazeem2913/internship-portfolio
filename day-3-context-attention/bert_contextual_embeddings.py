"""
bert_contextual_embeddings.py
--------------------------------
Pulls contextual embeddings for the Day 2 polysemous word ("light") from a
pretrained BERT model, in its two different-sense sentences. Shows that,
unlike Word2Vec (which returned an IDENTICAL vector both times), BERT
produces DIFFERENT vectors depending on context -- resolving Day 2's open
question.

REQUIRES INTERNET ACCESS to download the pretrained 'bert-base-uncased'
weights (~440MB) from HuggingFace on first run. Run this on your local
machine, not in a network-restricted sandbox.
"""

import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"

# The same two sentences from Day 2's polysemy_demo.py
SENTENCE_A = "They shed light on social and economic issues, including student debt and food insecurity"
SENTENCE_B = "A Sun-like star is about a billion times brighter than the reflected light from any exoplanet orbiting it"

WORD = "light"


def get_word_vector(sentence, word, tokenizer, model):
    """
    Returns BERT's final-layer contextual embedding for the FIRST occurrence
    of `word` in `sentence`. BERT uses WordPiece tokenization, so we find the
    token position(s) corresponding to our target word after tokenization.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state shape: [1, seq_len, 768]
    hidden_states = outputs.last_hidden_state[0]  # [seq_len, 768]

    # Map back from token ids to find which position(s) correspond to our word
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    target_positions = [i for i, tok in enumerate(tokens) if tok == word]

    if not target_positions:
        raise ValueError(f"Word '{word}' not found as a standalone token in: {tokens}")

    # If the word got split into subword pieces, this simple demo assumes a
    # whole-word match (true for common words like "light" in bert-base-uncased).
    position = target_positions[0]
    vector = hidden_states[position]
    return vector, tokens, position


if __name__ == "__main__":
    print(f"Loading {MODEL_NAME} (this downloads ~440MB on first run)...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    model.eval()

    lines = ["=" * 90, "BERT CONTEXTUAL EMBEDDINGS: resolving Day 2's open question", "=" * 90]

    lines.append(f'\nWord under test: "{WORD}"\n')
    lines.append(f'Sentence A (idiomatic sense - "reveal/inform"):\n  "{SENTENCE_A}"\n')
    lines.append(f'Sentence B (literal sense - "electromagnetic radiation"):\n  "{SENTENCE_B}"\n')

    vec_a, tokens_a, pos_a = get_word_vector(SENTENCE_A, WORD, tokenizer, model)
    vec_b, tokens_b, pos_b = get_word_vector(SENTENCE_B, WORD, tokenizer, model)

    lines.append(f"BERT tokens (Sentence A): {tokens_a}")
    lines.append(f"'{WORD}' found at position {pos_a}\n")
    lines.append(f"BERT tokens (Sentence B): {tokens_b}")
    lines.append(f"'{WORD}' found at position {pos_b}\n")

    lines.append(f"BERT vector for '{WORD}' in Sentence A (first 10 of 768 dims):")
    lines.append(f"  {vec_a[:10].tolist()}")
    lines.append(f"\nBERT vector for '{WORD}' in Sentence B (first 10 of 768 dims):")
    lines.append(f"  {vec_b[:10].tolist()}")

    are_identical = torch.equal(vec_a, vec_b)
    cos_sim = torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()

    lines.append(f"\nAre the two vectors IDENTICAL? {are_identical}")
    lines.append(f"Cosine similarity between the two contextual vectors: {cos_sim:.4f}")

    lines.append("\nCONCLUSION:")
    lines.append("Unlike Word2Vec -- which returned a BIT-FOR-BIT IDENTICAL vector for 'light'")
    lines.append("in both sentences (cosine similarity 1.000000, proven in Day 2) -- BERT")
    lines.append("produces genuinely DIFFERENT vectors for the same word string, because each")
    lines.append("occurrence's final representation is computed by self-attention looking at")
    lines.append("the ACTUAL surrounding words in that SPECIFIC sentence. A moderate-to-low")
    lines.append("cosine similarity here (rather than 1.0) is the expected, correct signal --")
    lines.append("it shows the model has represented these as meaningfully different senses")
    lines.append("of the same word, resolving exactly the limitation Day 2 ended on.")

    output = "\n".join(lines)
    print(output)

    with open("outputs/bert_contextual_results.txt", "w", encoding="utf-8") as f:
        f.write(output)

    # Save vectors for reuse in the comparison table and t-SNE scripts
    torch.save({"vec_a": vec_a, "vec_b": vec_b}, "outputs/bert_light_vectors.pt")

    print("\n\nSaved to outputs/bert_contextual_results.txt")
