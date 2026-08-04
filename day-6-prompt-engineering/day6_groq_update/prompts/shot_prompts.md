# Zero-Shot / One-Shot / Few-Shot Prompts

Templates used in zero_one_few_shot_comparison_openai.py, for the three
task types: classification (sarcasm-aware sentiment), extraction (invoice
fields), and generation (brand-voice product copy).

---

## CLASSIFICATION_ZERO_SHOT

Classify the sentiment of this review as Positive, Negative, or Neutral:
"{review}"

---

## CLASSIFICATION_ONE_SHOT

Classify the sentiment as Positive, Negative, or Neutral. Note that some
reviews are SARCASTIC -- classify based on the underlying intent, not
surface wording.

Review: "Wow, five stars, my order arrived a MONTH late. Just wonderful service."
Sentiment: Negative (sarcastic -- complaining about a late order)

Review: "{review}"
Sentiment:

---

## CLASSIFICATION_FEW_SHOT

Classify the sentiment as Positive, Negative, or Neutral. Note that some
reviews are SARCASTIC -- classify based on the underlying intent, not
surface wording.

Review: "Wow, five stars, my order arrived a MONTH late. Just wonderful service."
Sentiment: Negative (sarcastic -- complaining about a late order)

Review: "This blender is genuinely amazing, works perfectly every time."
Sentiment: Positive (sincere praise)

Review: "Love how the app crashes every time I open it. Truly a masterpiece."
Sentiment: Negative (sarcastic -- complaining about crashes)

Review: "{review}"
Sentiment:

---

## EXTRACTION_ZERO_SHOT

Extract the client name, due date, and amount from this invoice text:
"{invoice_text}"

---

## EXTRACTION_ONE_SHOT

Extract client name, due date (YYYY-MM-DD format, assume current year 2026
if no year given), and amount (as a plain number, no words) from invoice text.

Text: "Invoice 4471, client: Blue Horizon Ltd, due on the 15th of January,
total due: five hundred and twenty dollars"
Output: {{"name": "Blue Horizon Ltd", "due_date": "2026-01-15", "amount": 520}}

Text: "{invoice_text}"
Output:

---

## EXTRACTION_FEW_SHOT

Extract client name, due date (YYYY-MM-DD format, assume current year 2026
if no year given), and amount (as a plain number, no words) from invoice text.

Text: "Invoice 4471, client: Blue Horizon Ltd, due on the 15th of January,
total due: five hundred and twenty dollars"
Output: {{"name": "Blue Horizon Ltd", "due_date": "2026-01-15", "amount": 520}}

Text: "Ref #99120 for Riverside Design Studio, payment due 2nd Aug, owes
one thousand one hundred dollars"
Output: {{"name": "Riverside Design Studio", "due_date": "2026-08-02", "amount": 1100}}

Text: "{invoice_text}"
Output:

---

## GENERATION_ZERO_SHOT

Write a product description for: {product_description}

---

## GENERATION_ONE_SHOT

Write a product description matching this exact voice: terse, technical,
no marketing adjectives, spec-forward, short sentences.

Product: 65% mechanical keyboard, aluminum frame, USB-C.
Description: "65% layout. Aluminum frame. USB-C connection. No number pad.
Standard bottom row. Compatible with all Cherry MX-style switches."

Product: {product_description}
Description:

---

## GENERATION_FEW_SHOT

Write a product description matching this exact voice: terse, technical,
no marketing adjectives, spec-forward, short sentences.

Product: 65% mechanical keyboard, aluminum frame, USB-C.
Description: "65% layout. Aluminum frame. USB-C connection. No number pad.
Standard bottom row. Compatible with all Cherry MX-style switches."

Product: Wireless mouse, 3200 DPI, rechargeable.
Description: "3200 DPI optical sensor. Rechargeable via USB-C. Approx. 70
hours per charge. 6 programmable buttons. 95g weight."

Product: {product_description}
Description:
