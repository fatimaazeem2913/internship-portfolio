"""
generate_source_pdfs.py
--------------------------
Creates 3 real PDF documents from different domains, used as the source
corpus for Day 7's retrieval mini-project:
    1. A research-paper-style excerpt on Transformer attention mechanisms
    2. A news-article-style piece on Pakistan's presidency
    3. A technical-manual-style excerpt for a WiFi router setup guide

These are genuinely authored (not downloaded) but written in the authentic
register of each domain, and are processed exactly like real source
documents would be -- extracted, chunked, cleaned, and retrieved from.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

styles = getSampleStyleSheet()
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11, leading=15, spaceAfter=10)
heading_style = styles["Heading1"]
subheading_style = styles["Heading2"]


def build_pdf(filename, title, sections):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                             topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    story = [Paragraph(title, heading_style), Spacer(1, 12)]
    for heading, paragraphs in sections:
        if heading:
            story.append(Paragraph(heading, subheading_style))
            story.append(Spacer(1, 6))
        for p in paragraphs:
            story.append(Paragraph(p, body_style))
    doc.build(story)
    print(f"Created {filename}")


# ===================================================================
# DOCUMENT 1: RESEARCH PAPER -- Transformer attention mechanisms
# ===================================================================
RESEARCH_PAPER_SECTIONS = [
    ("Abstract", [
        "Self-attention has become the dominant mechanism for sequence modeling "
        "in natural language processing, replacing recurrent architectures in "
        "most modern systems. This paper reviews the scaled dot-product attention "
        "mechanism, its computational properties, and the architectural choices "
        "that allow Transformer models to be trained in parallel across very long "
        "sequences.",
    ]),
    ("1. Introduction", [
        "Recurrent neural networks process sequences one token at a time, with "
        "each hidden state depending on the previous one. This sequential "
        "dependency prevents parallelization during training and causes "
        "gradients to vanish across long sequences, since the gradient signal "
        "must pass through many multiplicative steps to reach early tokens.",
        "Self-attention resolves both limitations by allowing every position in "
        "a sequence to attend directly to every other position in a single "
        "operation, with a path length of one between any two tokens regardless "
        "of their distance in the sequence.",
    ]),
    ("2. Scaled Dot-Product Attention", [
        "Given a set of queries Q, keys K, and values V, attention is computed "
        "as the softmax of the scaled dot product between queries and keys, "
        "applied to the values. The scaling factor of one over the square root "
        "of the key dimension prevents the dot products from growing too large "
        "in magnitude, which would otherwise push the softmax function into "
        "regions with extremely small gradients.",
        "Multi-head attention extends this mechanism by computing several "
        "attention operations in parallel, each over a smaller subspace of the "
        "representation. This allows different heads to specialize in "
        "different types of relationships between tokens, such as syntactic "
        "dependencies or coreference.",
    ]),
    ("3. Positional Encoding", [
        "Because self-attention has no inherent notion of token order, position "
        "information must be injected separately. The original Transformer uses "
        "sinusoidal functions of different frequencies for this purpose, which "
        "allows the model to generalize to sequence lengths not seen during "
        "training and to learn relative-position relationships as linear "
        "transformations.",
    ]),
    ("4. Conclusion", [
        "The combination of self-attention, position-wise feed-forward "
        "networks, residual connections, and layer normalization allows "
        "Transformer models to be trained efficiently at very large scale, and "
        "has become the foundation of essentially all modern large language "
        "models.",
    ]),
]

# ===================================================================
# DOCUMENT 2: NEWS ARTICLE -- Pakistan's presidency
# ===================================================================
NEWS_ARTICLE_SECTIONS = [
    (None, [
        "ISLAMABAD -- Asif Ali Zardari is the current President of Pakistan, "
        "having assumed office as the 14th President on 10 March 2024. This is "
        "his second term in the role, having previously served as the 11th "
        "President of Pakistan from September 2008 to September 2013.",
        "Zardari's re-election makes him the first civilian in Pakistan's "
        "history to serve two non-consecutive presidential terms. He succeeded "
        "Arif Alvi, who served as the 13th President from September 2018 to "
        "March 2024.",
        "As President, Zardari serves as the ceremonial head of state and "
        "ceremonial commander-in-chief of the Pakistan Armed Forces, while "
        "executive authority in Pakistan's parliamentary system rests "
        "primarily with the Prime Minister, currently Shehbaz Sharif. The "
        "President's official residence is the Aiwan-e-Sadr in Islamabad.",
        "Zardari currently heads the Pakistan Peoples Party, a position he has "
        "held since December 2015. His political career has spanned several "
        "decades, including serving as a federal minister in the 1990s under "
        "Prime Minister Benazir Bhutto, to whom he was married.",
        "Under Pakistan's constitution, the President is elected by an "
        "electoral college for a term of five years, renewable once "
        "consecutively. The office was first established in 1956, with "
        "Iskander Mirza serving as its inaugural holder.",
    ]),
]

# ===================================================================
# DOCUMENT 3: TECHNICAL MANUAL -- WiFi router setup guide
# ===================================================================
TECH_MANUAL_SECTIONS = [
    ("1. Package Contents", [
        "Your router package includes: one dual-band wireless router, one "
        "power adapter (12V, 1.5A), one Ethernet cable, and this quick start "
        "guide. Verify all items are present before proceeding with setup.",
    ]),
    ("2. Initial Setup", [
        "Connect the power adapter to the router and to a wall outlet. Wait "
        "approximately 60 seconds for the power LED to turn solid green, "
        "indicating the router has finished booting. Connect one end of the "
        "included Ethernet cable to the WAN port on the router, and the other "
        "end to your modem.",
        "Using a computer or mobile device, connect to the default wireless "
        "network printed on the label on the bottom of the router. Open a web "
        "browser and navigate to 192.168.1.1 to access the setup wizard.",
    ]),
    ("3. Configuring Wireless Settings", [
        "In the setup wizard, you will be prompted to create a new network "
        "name (SSID) and password. We recommend using WPA3 encryption if your "
        "devices support it, or WPA2 as a fallback. Avoid using WEP encryption, "
        "as it is no longer considered secure.",
        "The router supports simultaneous dual-band operation on both the "
        "2.4GHz and 5GHz frequencies. The 2.4GHz band offers longer range but "
        "lower maximum speeds; the 5GHz band offers higher speeds but shorter "
        "range and is more susceptible to interference from walls.",
    ]),
    ("4. Troubleshooting", [
        "If the power LED is blinking amber, the router is experiencing a "
        "firmware issue and may need to be reset. Press and hold the reset "
        "button on the back panel for 10 seconds to restore factory default "
        "settings. Note that this will erase any custom configuration.",
        "If devices cannot connect to the internet despite a strong WiFi "
        "signal, verify the WAN cable is securely connected to the modem, and "
        "confirm the modem itself has an active internet connection by "
        "checking its own status lights.",
    ]),
]


if __name__ == "__main__":
    build_pdf("pdfs/research_paper.pdf",
              "Scaled Dot-Product Attention: A Review",
              RESEARCH_PAPER_SECTIONS)
    build_pdf("pdfs/news_article.pdf",
              "Zardari Serving Second Term as President of Pakistan",
              NEWS_ARTICLE_SECTIONS)
    build_pdf("pdfs/technical_manual.pdf",
              "WiFi Router Quick Start Guide",
              TECH_MANUAL_SECTIONS)
