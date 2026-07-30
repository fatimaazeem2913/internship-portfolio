"""
scraper.py
----------
Scrapes raw text content from 3 different domain web pages (news, science, dialogue/forum)
and writes it into data/input_corpus.txt, tagged by domain.
"""

import requests
from bs4 import BeautifulSoup
import time

SOURCES = {
  "NEWS": [
        "https://en.wikipedia.org/wiki/COVID-19_pandemic",
    ],
    "SCIENCE": [
        "https://en.wikipedia.org/wiki/Exoplanet",
    ],
 "DIALOGUE": [
        "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do",
    ],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NLP-Learning-Bot/1.0; educational use)"
}


def fetch_page_text(url):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    paragraphs = soup.find_all(["p", "h1", "h2", "h3"])
    text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return text


def build_corpus(output_path="data/input_corpus.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for domain, urls in SOURCES.items():
            f.write(f"### DOMAIN: {domain} ###\n")
            for url in urls:
                try:
                    print(f"Fetching [{domain}] {url}")
                    text = fetch_page_text(url)
                    f.write(text + "\n")
                except Exception as e:
                    print(f"  Failed to fetch {url}: {e}")
                time.sleep(1)
            f.write("\n")
    print(f"\nCorpus written to {output_path}")


if __name__ == "__main__":
    build_corpus()