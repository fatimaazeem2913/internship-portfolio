"""One-off script: creates a real scanned-style PDF with NO extractable text
layer -- i.e. genuinely just an image of text, rendered to look like a
photocopied/scanned page -- so pytesseract has something real to OCR, and
native PDF text extraction genuinely fails on it (proving the comparison
is real, not staged)."""

from PIL import Image, ImageDraw, ImageFont
import img2pdf

WIDTH, HEIGHT = 1700, 2200  # roughly a 200-DPI Letter page

TEXT_LINES = [
    "ACME CORPORATION",
    "Vendor Non-Disclosure Agreement",
    "",
    "This Non-Disclosure Agreement (\"Agreement\") is entered into as of the",
    "date of last signature below, by and between Acme Corporation",
    "(\"Company\") and the vendor identified in the signature block below",
    "(\"Vendor\"), collectively the \"Parties.\"",
    "",
    "1. CONFIDENTIAL INFORMATION",
    "",
    "For purposes of this Agreement, \"Confidential Information\" means any",
    "non-public technical, business, or financial information disclosed by",
    "either Party, whether disclosed orally, in writing, or by inspection of",
    "tangible items, that is designated as confidential at the time of",
    "disclosure or that a reasonable person would understand to be",
    "confidential given the nature of the information and circumstances of",
    "disclosure.",
    "",
    "2. OBLIGATIONS OF RECEIVING PARTY",
    "",
    "The receiving Party agrees to hold the disclosing Party's Confidential",
    "Information in strict confidence and not to disclose such Confidential",
    "Information to any third party without prior written consent, except to",
    "employees and contractors with a legitimate need to know, who are",
    "themselves bound by confidentiality obligations at least as protective",
    "as those in this Agreement.",
    "",
    "3. TERM",
    "",
    "This Agreement shall remain in effect for a period of three (3) years",
    "from the Effective Date, provided that obligations with respect to any",
    "Confidential Information disclosed during the term shall survive",
    "termination of this Agreement for an additional period of two (2)",
    "years thereafter.",
    "",
    "4. EXCLUSIONS",
    "",
    "This Agreement imposes no obligation with respect to information that:",
    "(a) was already known to the receiving Party prior to disclosure;",
    "(b) becomes publicly available through no fault of the receiving Party;",
    "(c) is independently developed without reference to the Confidential",
    "Information; or (d) is rightfully received from a third party without",
    "restriction.",
]

img = Image.new("L", (WIDTH, HEIGHT), color=255)
draw = ImageDraw.Draw(img)

try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
except Exception:
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

y = 100
for i, line in enumerate(TEXT_LINES):
    font = font_title if i == 0 else font_body
    draw.text((100, y), line, fill=0, font=font)
    y += 42 if line else 30

# Add a very light noise/texture pass to make this look more like a genuine
# scan rather than crisp rendered text, and to give OCR something slightly
# imperfect to actually contend with (a completely pristine render makes
# for an unfairly easy OCR comparison).
import random
random.seed(42)
pixels = img.load()
for _ in range(60000):
    x = random.randint(0, WIDTH - 1)
    yy = random.randint(0, HEIGHT - 1)
    if pixels[x, yy] == 255:
        pixels[x, yy] = random.randint(230, 250)

img_path = "data/scanned/scanned_page_raw.png"
img.save(img_path)

pdf_path = "data/scanned/vendor_nda_scanned.pdf"
with open(pdf_path, "wb") as f:
    f.write(img2pdf.convert(img_path))

print(f"Scanned-style PDF generated: {pdf_path}")
