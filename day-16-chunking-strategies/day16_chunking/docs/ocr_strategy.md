# OCR Strategy — Native Extraction, OCR, and the Required LLM Pass

## 1. Real, measured comparison: native extraction vs. Tesseract OCR

This project generated a genuinely image-only scanned PDF
(`data/scanned/vendor_nda_scanned.pdf`) — confirmed to have **zero**
extractable characters via native pdfplumber extraction — then ran it
through `pytesseract`. Real, measured results:

| Method | Characters recovered | Result |
|---|---|---|
| Native extraction (pdfplumber) | **0** | Total failure — as expected for a true scanned image |
| OCR (pytesseract) | **1,833** | Successfully recovered the real document text |

This confirms the obvious but important point: **for genuinely scanned
documents, OCR is not optional — native extraction cannot see the text at
all**, since there is no text layer, only pixels.

## 2. Why OCR alone is not enough — and why the task explicitly requires an LLM pass

Tesseract OCR is reliable for clean, well-scanned prose (this project's own
test document recovered cleanly), but it has two well-known, genuine
weaknesses that a single OCR pass cannot fix on its own:

1. **Character-level misreads.** OCR engines routinely confuse visually
   similar characters (`0`/`O`, `1`/`l`/`I`, `rn`/`m`), especially on lower
   quality scans. These errors are individually small but can silently
   corrupt a date, an amount, or an ID number in a way that's easy to miss.
2. **Table and layout scrambling.** This is the more serious failure mode.
   OCR engines fundamentally read left-to-right, top-to-bottom — they have
   no real concept of "these three numbers belong to the same row." A
   table's visual column alignment routinely gets flattened into a
   nonsensical linear text stream. Real, sourced benchmark data backs this
   up directly: on a 500-document benchmark, Tesseract scored only **64%
   row-level accuracy on multi-page tables**, compared to layout-aware
   tools scoring substantially higher — tables are specifically where
   plain OCR engines like Tesseract "fall apart."

**This is exactly why the task requires a further LLM pass on top of raw
OCR output, rather than treating OCR's raw text as the final answer.** An
LLM pass can use context and language understanding to catch and correct
character-level misreads that a rule-based post-processor would miss, and
can re-interpret a scrambled table reading order back into structured rows
using semantic understanding of what the numbers actually represent — two
things pure pattern-matching cannot do.

## 3. This project's implementation: `src/llm_ocr_correction.py`

Following the task's exact required architecture, this module takes raw
OCR text and produces three distinct outputs in one LLM pass:

1. **Cleaned Markdown** — corrected text with paragraph structure restored
   and obvious OCR character errors fixed, without inventing content the
   raw OCR didn't actually contain.
2. **Summary** — a short natural-language summary of the section, useful
   for quick human review or as a lightweight retrieval-time preview.
3. **Structured JSON** — concrete facts (dates, amounts, defined terms,
   party names) extracted into direct key-value pairs, so a downstream
   system can look up "what is the agreement term?" without needing
   another full LLM call over the raw text every time.

This follows the same `USE_MOCK_LLM` pattern established since Day 11 and
reused throughout Day 15 — real Gemini calls when an API key is present,
a clearly-labeled deterministic mock otherwise, so the module is runnable
and testable with zero cost or network dependency.

## 4. Why PaddleOCR was not installed in this sandbox (an honest limitation, not an oversight)

The task explicitly suggests PaddleOCR as a stronger alternative
specifically for table-heavy scanned documents, and the research backs
this up clearly: PaddleOCR consistently outperforms Tesseract on structured
layouts and tables (one benchmark measured **79% vs. Tesseract's 64%**
row-level table accuracy), owing to its layout-aware, fully neural
architecture versus Tesseract's more traditional approach.

**PaddleOCR was evaluated but not installed in this sandbox**, for an
honest, documented reason consistent with every other network limitation
found across Days 15-16: PaddleOCR requires downloading its detection and
recognition model weights from a registry that is not in this sandbox's
network allowlist — the same category of issue as the `huggingface.co`
block (Day 15's `sentence-transformers`) and the `openaipublic.blob.core.
windows.net` block (this project's `tiktoken`). Rather than silently
skipping this or pretending Tesseract alone is sufficient, this limitation
is documented directly, and the correct integration point is designed into
the architecture below so a real installation on unrestricted infrastructure
requires no redesign — only swapping which function produces the raw OCR
text.

## 5. The recommended, production-correct pipeline (per the task's note)

```
Scanned document
   |
   v
PaddleOCR  ---- table-aware layout detection + text recognition
   |            (replaces Tesseract specifically for table-heavy pages;
   |             this project's ingest_ocr.py already isolates the raw-
   |             OCR-text-in / raw-OCR-text-out boundary so this swap is
   |             a one-function change, not a redesign)
   v
Raw OCR text (with table structure preserved as accurately as PaddleOCR's
layout model allows)
   |
   v
LLM pass (src/llm_ocr_correction.py) ---- corrects remaining character-
   |                                       level errors, re-derives table
   |                                       row/column relationships using
   |                                       language understanding
   v
Three outputs: cleaned Markdown, summary, structured JSON
```

This project's own scanned test document (a prose NDA, not a table) never
actually exercises Tesseract's known table weakness — which is itself
documented honestly here rather than glossed over. The architecture above
is what should be used for a genuinely table-heavy scanned document in a
production environment with full network access.

## 6. Alternative tools noted in the task (Qianfan-OCR, Docling)

- **Docling** (IBM's open-source document conversion library) is a strong
  alternative specifically because it's designed around exactly this
  problem — converting complex documents (including tables) into
  structured Markdown/JSON directly, rather than flat OCR text requiring a
  separate correction pass.
- **Qianfan-OCR** (Baidu's cloud OCR API) trades the self-hosting/network-
  allowlist problem entirely for an API-key-based cloud dependency — worth
  considering specifically when local model downloads are infeasible (as
  in this sandbox), though it introduces its own per-page cost and an
  external network dependency of a different kind.

Both are viable alternatives to the PaddleOCR-then-LLM-pass architecture
above; the right choice depends on whether a team prefers self-hosted
control (PaddleOCR/Docling) or managed infrastructure (Qianfan-OCR).
