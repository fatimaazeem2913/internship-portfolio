"""
bert_synonym_embeddings.py
-----------------------------
Computes BERT contextual embeddings for the same 5 synonym word pairs tested
with TF-IDF (Day 2) and Word2Vec (Day 2), for the Day 3 comparison table.

REQUIRES INTERNET ACCESS to download 'bert-base-uncased' (run locally).
"""

import json
import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"

# Each word embedded in a simple, neutral carrier sentence so BERT has
# SOME context to work with (BERT embeddings are context-dependent by design,
# so a bare single word isn't a natural input).
SYNONYM_PAIRS = [
    ("virus", "pathogen", "The virus spread quickly.", "The pathogen spread quickly."),
    ("disease", "illness", "Doctors studied the disease.", "Doctors studied the illness."),
    ("planet", "world", "They discovered a new planet.", "They discovered a new world."),
    ("study", "research", "She began her study on the topic.", "She began her research on the topic."),
    ("function", "method", "The function returns a value.", "The method returns a value."),
]


def get_word_vector(sentence, word, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    target_positions = [i for i, tok in enumerate(tokens) if tok == word.lower()]
    if not target_positions:
        return None
    return hidden_states[target_positions[0]]


if __name__ == "__main__":
    print(f"Loading {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    model.eval()

    lines = ["=" * 90, "BERT SYNONYM PAIR SIMILARITY (for Day 3 comparison table)", "=" * 90, ""]
    results = {}

    for w1, w2, sent1, sent2 in SYNONYM_PAIRS:
        v1 = get_word_vector(sent1, w1, tokenizer, model)
        v2 = get_word_vector(sent2, w2, tokenizer, model)
        if v1 is None or v2 is None:
            lines.append(f"{w1} / {w2}: N/A (subword tokenization mismatch, see tokens manually)")
            results[f"{w1}/{w2}"] = None
            continue
        sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        results[f"{w1}/{w2}"] = sim
        lines.append(f"{w1} / {w2}: cosine similarity = {sim:.4f}")

    output = "\n".join(lines)
    print(output)

    with open("outputs/bert_synonym_results.txt", "w", encoding="utf-8") as f:
        f.write(output)
    with open("outputs/bert_synonym_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n\nSaved to outputs/bert_synonym_results.txt and .json")
