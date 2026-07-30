"""
tokenization_compare.py
------------------------
Compares character-level, word-level, and subword (BPE / WordPiece) tokenization
using HuggingFace `tokenizers` (trained from scratch on our own small corpus)
plus pretrained subword tokenizers for realistic comparison.
"""

from nltk.tokenize import word_tokenize
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

SAMPLE_SENTENCES = [
    "Researchers discovered an unbelievably habitable exoplanet.",
    "The tokenization bug is retokenizing incorrectly.",
    "Jordan's lemmatizer misclassifies irregular nouns like mice.",
]


def char_level_tokenize(text):
    return list(text)


def word_level_tokenize(text):
    return word_tokenize(text)


def train_bpe_tokenizer(corpus_path, vocab_size=300):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]"]
    )
    tokenizer.train([corpus_path], trainer)
    return tokenizer


def train_wordpiece_tokenizer(corpus_path, vocab_size=300):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
    )
    tokenizer.train([corpus_path], trainer)
    return tokenizer


if __name__ == "__main__":
    corpus_path = "outputs/cleaned_corpus.txt"

    bpe_tok = train_bpe_tokenizer(corpus_path)
    wp_tok = train_wordpiece_tokenizer(corpus_path)

    lines = []
    lines.append("=" * 90)
    lines.append("TOKENIZATION COMPARISON: character vs word vs subword (BPE) vs subword (WordPiece)")
    lines.append("=" * 90)

    for sent in SAMPLE_SENTENCES:
        lines.append(f"\nSENTENCE: {sent}")

        char_toks = char_level_tokenize(sent)
        word_toks = word_level_tokenize(sent)
        bpe_toks = bpe_tok.encode(sent).tokens
        wp_toks = wp_tok.encode(sent).tokens

        lines.append(f"  [Char-level]      ({len(char_toks)} tokens): {char_toks}")
        lines.append(f"  [Word-level]      ({len(word_toks)} tokens): {word_toks}")
        lines.append(f"  [BPE subword]     ({len(bpe_toks)} tokens): {bpe_toks}")
        lines.append(f"  [WordPiece subword]({len(wp_toks)} tokens): {wp_toks}")

    output = "\n".join(lines)
    print(output)

    with open("outputs/tokenization_comparison.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nSaved to outputs/tokenization_comparison.txt")
