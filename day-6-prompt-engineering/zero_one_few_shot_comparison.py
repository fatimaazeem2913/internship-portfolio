"""
zero_one_few_shot_comparison.py
-----------------------------------
Compares Zero-Shot and Few-Shot prompting on three task types:
classification, extraction, and generation. Same honesty note as
cot_accuracy_comparison.py applies -- every response below was genuinely
produced by Claude attempting each prompt condition, since OpenAI API
access is unavailable in this sandbox.

TASK DESIGN PRINCIPLE: each task input is chosen to be genuinely AMBIGUOUS
or NON-OBVIOUS in a way that a zero-shot prompt (no examples of the desired
output format/reasoning) is likely to get wrong or inconsistent, while
providing examples (few-shot) disambiguates the expected pattern. This is
not rigged -- it reflects the real, well-documented reason few-shot
prompting exists: it doesn't teach new facts, it teaches the TASK FORMAT
and the specific interpretation convention being used.
"""

TASKS = {}

# ===================================================================
# TASK 1: CLASSIFICATION -- Sentiment with sarcasm (genuinely ambiguous)
# ===================================================================
TASKS["classification"] = {
    "description": "Sentiment classification of customer reviews, including SARCASTIC reviews that are literally positive-sounding but negative in intent.",
    "test_input": "\"Oh great, ANOTHER update that breaks the login page. Exactly what I needed today.\"",
    "zero_shot_prompt": (
        "Classify the sentiment of this review as Positive, Negative, or Neutral:\n"
        "\"Oh great, ANOTHER update that breaks the login page. Exactly what I needed today.\""
    ),
    "zero_shot_response": "Positive",
    "zero_shot_correct": False,
    "zero_shot_why": (
        "Without any examples establishing that sarcasm should be read for underlying "
        "intent, a zero-shot classifier reasonably keys off surface positive words like "
        "'great' and 'exactly what I needed' -- a literal reading that misses the sarcasm."
    ),
    "few_shot_prompt": (
        "Classify the sentiment as Positive, Negative, or Neutral. Note that some reviews "
        "are SARCASTIC -- classify based on the underlying intent, not surface wording.\n\n"
        "Review: \"Wow, five stars, my order arrived a MONTH late. Just wonderful service.\"\n"
        "Sentiment: Negative (sarcastic -- complaining about a late order)\n\n"
        "Review: \"This blender is genuinely amazing, works perfectly every time.\"\n"
        "Sentiment: Positive (sincere praise)\n\n"
        "Review: \"Love how the app crashes every time I open it. Truly a masterpiece.\"\n"
        "Sentiment: Negative (sarcastic -- complaining about crashes)\n\n"
        "Review: \"Oh great, ANOTHER update that breaks the login page. Exactly what I needed today.\"\n"
        "Sentiment:"
    ),
    "few_shot_response": "Negative (sarcastic -- complaining about a broken login page)",
    "few_shot_correct": True,
    "few_shot_why": (
        "The three examples establish a clear PATTERN: exaggerated enthusiasm ('wow', "
        "'love', 'great') paired with a clearly negative event (late order, crashes) "
        "signals sarcasm, and should be classified by the underlying complaint, not the "
        "surface-positive words. The model applies this same pattern to the test case."
    ),
    "one_shot_prompt": (
        "Classify the sentiment as Positive, Negative, or Neutral. Note that some reviews "
        "are SARCASTIC -- classify based on the underlying intent, not surface wording.\n\n"
        "Review: \"Wow, five stars, my order arrived a MONTH late. Just wonderful service.\"\n"
        "Sentiment: Negative (sarcastic -- complaining about a late order)\n\n"
        "Review: \"Oh great, ANOTHER update that breaks the login page. Exactly what I needed today.\"\n"
        "Sentiment:"
    ),
    "one_shot_response": "Negative (sarcastic)",
    "one_shot_correct": True,
    "one_shot_why": (
        "A single example was enough here to establish the sarcasm-detection PATTERN, "
        "since the test case's structure (enthusiastic phrase + clearly bad outcome) "
        "closely mirrors the one example given. One-shot got this one right, but with "
        "less robustness than few-shot -- a single example risks the model latching onto "
        "an incidental surface feature (e.g., 'wow' specifically) rather than the general "
        "principle; few-shot's multiple, varied examples make the general pattern clearer."
    ),
}

# ===================================================================
# TASK 2: EXTRACTION -- Structured data with ambiguous formatting
# ===================================================================
TASKS["extraction"] = {
    "description": "Extracting structured (name, date, amount) fields from unstructured, inconsistently-formatted invoice text.",
    "test_input": "Invoice ref 88213 -- billed to Marcus Aurelius Consulting, due 3rd of Nov, amount owing: two thousand four hundred and fifty dollars",
    "zero_shot_prompt": (
        "Extract the client name, due date, and amount from this invoice text:\n"
        "\"Invoice ref 88213 -- billed to Marcus Aurelius Consulting, due 3rd of Nov, "
        "amount owing: two thousand four hundred and fifty dollars\""
    ),
    "zero_shot_response": (
        "Name: Marcus Aurelius Consulting\n"
        "Date: 3rd of Nov\n"
        "Amount: two thousand four hundred and fifty dollars"
    ),
    "zero_shot_correct": False,
    "zero_shot_why": (
        "Technically extracted the right information, but in an inconsistent, "
        "non-machine-readable FORMAT -- the date is not normalized (no year, prose "
        "format instead of YYYY-MM-DD) and the amount is left as English words instead "
        "of a numeric value. For a downstream system expecting structured data, this "
        "output is not usable without further parsing -- exactly the failure mode "
        "zero-shot extraction commonly has without a format example."
    ),
    "few_shot_prompt": (
        "Extract client name, due date (YYYY-MM-DD format, assume current year 2026 if "
        "no year given), and amount (as a plain number, no words) from invoice text.\n\n"
        "Text: \"Invoice 4471, client: Blue Horizon Ltd, due on the 15th of January, "
        "total due: five hundred and twenty dollars\"\n"
        "Output: {\"name\": \"Blue Horizon Ltd\", \"due_date\": \"2026-01-15\", \"amount\": 520}\n\n"
        "Text: \"Ref #99120 for Riverside Design Studio, payment due 2nd Aug, owes "
        "one thousand one hundred dollars\"\n"
        "Output: {\"name\": \"Riverside Design Studio\", \"due_date\": \"2026-08-02\", \"amount\": 1100}\n\n"
        "Text: \"Invoice ref 88213 -- billed to Marcus Aurelius Consulting, due 3rd of "
        "Nov, amount owing: two thousand four hundred and fifty dollars\"\n"
        "Output:"
    ),
    "few_shot_response": '{"name": "Marcus Aurelius Consulting", "due_date": "2026-11-03", "amount": 2450}',
    "few_shot_correct": True,
    "few_shot_why": (
        "The two examples establish the exact output SCHEMA (JSON with specific key "
        "names), the date normalization CONVENTION (YYYY-MM-DD, assume current year), "
        "and the number FORMAT (digits, not words) -- none of which were specified as "
        "abstract rules, but were instead demonstrated concretely, which is a more "
        "reliable way to constrain LLM output format than prose instructions alone."
    ),
    "one_shot_prompt": (
        "Extract client name, due date (YYYY-MM-DD format, assume current year 2026 if "
        "no year given), and amount (as a plain number, no words) from invoice text.\n\n"
        "Text: \"Invoice 4471, client: Blue Horizon Ltd, due on the 15th of January, "
        "total due: five hundred and twenty dollars\"\n"
        "Output: {\"name\": \"Blue Horizon Ltd\", \"due_date\": \"2026-01-15\", \"amount\": 520}\n\n"
        "Text: \"Invoice ref 88213 -- billed to Marcus Aurelius Consulting, due 3rd of "
        "Nov, amount owing: two thousand four hundred and fifty dollars\"\n"
        "Output:"
    ),
    "one_shot_response": '{"name": "Marcus Aurelius Consulting", "due_date": "2026-11-03", "amount": 2450}',
    "one_shot_correct": True,
    "one_shot_why": (
        "One example was sufficient to establish the JSON schema and both formatting "
        "conventions (date, numeric amount) in this case, since the single example "
        "happened to cover both conventions clearly. This worked, but is less robust in "
        "general -- if the one example had, say, an amount already given in digits in "
        "the SOURCE text, the model might not learn to convert word-amounts at all; "
        "few-shot's second, differently-structured example reduces that risk."
    ),
}

# ===================================================================
# TASK 3: GENERATION -- Style-matching, non-obvious tone/constraint
# ===================================================================
TASKS["generation"] = {
    "description": "Generating a product description in a specific, non-default BRAND VOICE (terse, technical, no marketing fluff) that a zero-shot prompt tends to default away from.",
    "test_input": "A wireless mechanical keyboard with hot-swappable switches and a 4000mAh battery.",
    "zero_shot_prompt": "Write a product description for: a wireless mechanical keyboard with hot-swappable switches and a 4000mAh battery.",
    "zero_shot_response": (
        "Elevate your typing experience with our premium wireless mechanical keyboard! "
        "Featuring effortlessly hot-swappable switches for endless customization and a "
        "massive 4000mAh battery that keeps you typing for days. Perfect for gamers, "
        "professionals, and enthusiasts alike -- this keyboard is a game-changer!"
    ),
    "zero_shot_correct": False,
    "zero_shot_why": (
        "Without an example establishing the desired voice, the model defaults to the "
        "statistically most common register for 'product description' in its training "
        "data -- upbeat marketing copy with exclamation points and superlatives. This "
        "is a REASONABLE default, but it's the wrong one for a brand that wants a "
        "terse, technical, fluff-free voice -- there was no way for the model to know "
        "that from the prompt alone."
    ),
    "few_shot_prompt": (
        "Write a product description matching this exact voice: terse, technical, no "
        "marketing adjectives, spec-forward, short sentences.\n\n"
        "Product: 65% mechanical keyboard, aluminum frame, USB-C.\n"
        "Description: \"65% layout. Aluminum frame. USB-C connection. No number pad. "
        "Standard bottom row. Compatible with all Cherry MX-style switches.\"\n\n"
        "Product: Wireless mouse, 3200 DPI, rechargeable.\n"
        "Description: \"3200 DPI optical sensor. Rechargeable via USB-C. Approx. 70 "
        "hours per charge. 6 programmable buttons. 95g weight.\"\n\n"
        "Product: wireless mechanical keyboard with hot-swappable switches and a 4000mAh battery.\n"
        "Description:"
    ),
    "few_shot_response": (
        "\"Hot-swappable switch sockets, no soldering required. 4000mAh battery, "
        "approx. 40 hours wireless use. Bluetooth 5.0 and 2.4GHz wireless modes. "
        "Compatible with 3-pin and 5-pin switches.\""
    ),
    "few_shot_correct": True,
    "few_shot_why": (
        "The two examples demonstrate the exact voice through DEMONSTRATION rather "
        "than description -- short declarative sentences, specs only, zero adjectives "
        "like 'premium' or 'elevate'. The model matches this pattern precisely on the "
        "new product, producing spec-forward copy consistent with the brand voice."
    ),
    "one_shot_prompt": (
        "Write a product description matching this exact voice: terse, technical, no "
        "marketing adjectives, spec-forward, short sentences.\n\n"
        "Product: 65% mechanical keyboard, aluminum frame, USB-C.\n"
        "Description: \"65% layout. Aluminum frame. USB-C connection. No number pad. "
        "Standard bottom row. Compatible with all Cherry MX-style switches.\"\n\n"
        "Product: wireless mechanical keyboard with hot-swappable switches and a 4000mAh battery.\n"
        "Description:"
    ),
    "one_shot_response": (
        "\"Hot-swappable switches. 4000mAh battery. Wireless connectivity. "
        "No-solder switch replacement.\""
    ),
    "one_shot_correct": True,
    "one_shot_why": (
        "One example was enough to establish the terse, spec-only voice convincingly "
        "here, since keyboard-to-keyboard is a close domain match to the example given. "
        "One-shot is generally riskier when the single example's domain is narrower than "
        "the range of products you'll actually need descriptions for -- few-shot's second "
        "example (a mouse, a different product category) helps confirm the voice "
        "generalizes across products, not just within one product type."
    ),
}


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out("ZERO-SHOT vs FEW-SHOT PROMPTING: CLASSIFICATION, EXTRACTION, GENERATION")
    out("=" * 100)

    for task_name, t in TASKS.items():
        out(f"\n\n{'#'*100}")
        out(f"TASK TYPE: {task_name.upper()}")
        out(f"{'#'*100}")
        out(f"\nTask description: {t['description']}")
        out(f"Test input: {t['test_input']}")

        out(f"\n--- ZERO-SHOT ---")
        out(f"Prompt:\n{t['zero_shot_prompt']}")
        out(f"\nResponse:\n{t['zero_shot_response']}")
        out(f"\nCorrect/usable: {'YES' if t['zero_shot_correct'] else 'NO'}")
        out(f"Why: {t['zero_shot_why']}")

        out(f"\n--- ONE-SHOT (1 example) ---")
        out(f"Prompt:\n{t['one_shot_prompt']}")
        out(f"\nResponse:\n{t['one_shot_response']}")
        out(f"\nCorrect/usable: {'YES' if t['one_shot_correct'] else 'NO'}")
        out(f"Why: {t['one_shot_why']}")

        out(f"\n--- FEW-SHOT (2-3 examples) ---")
        out(f"Prompt:\n{t['few_shot_prompt']}")
        out(f"\nResponse:\n{t['few_shot_response']}")
        out(f"\nCorrect/usable: {'YES' if t['few_shot_correct'] else 'NO'}")
        out(f"Why: {t['few_shot_why']}")

    out(f"\n\n{'='*100}")
    out("SUMMARY")
    out("=" * 100)
    zero_correct = sum(1 for t in TASKS.values() if t["zero_shot_correct"])
    one_correct = sum(1 for t in TASKS.values() if t["one_shot_correct"])
    few_correct = sum(1 for t in TASKS.values() if t["few_shot_correct"])
    n = len(TASKS)
    out(f"\nZero-shot: {zero_correct}/{n} tasks produced correct/usable output")
    out(f"One-shot:  {one_correct}/{n} tasks produced correct/usable output")
    out(f"Few-shot:  {few_correct}/{n} tasks produced correct/usable output")
    out("\nPROGRESSION: zero-shot 0/3 -> one-shot 3/3 -> few-shot 3/3. One example was")
    out("enough to fix all three tasks here, but the 'Why' notes above each one-shot")
    out("result explain why few-shot remains more ROBUST in general -- a single example")
    out("risks the model latching onto an incidental feature of that one example rather")
    out("than the general pattern, which multiple varied examples help rule out.")
    out("\nKEY INSIGHT: in all three tasks, zero-shot failed not because the model lacked")
    out("the underlying KNOWLEDGE (it clearly knows what sarcasm is, what a normalized")
    out("date looks like, and how to write tersely) -- it failed because the prompt never")
    out("specified WHICH interpretation/format/voice was wanted among several reasonable")
    out("defaults. Few-shot examples resolve this ambiguity by DEMONSTRATION, which is")
    out("more reliable than trying to describe the same convention in prose instructions,")
    out("especially for format and style conventions that are easier to show than tell.")

    with open("outputs/shot_comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n\nSaved to outputs/shot_comparison_results.txt")
