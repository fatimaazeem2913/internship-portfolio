import os
import re
from typing import List, Optional, Dict, Any
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import pymupdf

from .models import DocumentElement

def ingest_native_pdf(file_path: str) -> List[DocumentElement]:
    """Extract text, headings, and tables from native PDF."""
    elements = []
    current_heading = "Document Root"
    if not os.path.exists(file_path):
        return elements

    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=False) or ""
            tables = page.extract_tables()
            table_md = ""
            if tables:
                for table in tables:
                    clean_table = [[c if c is not None else "" for c in row] for row in table]
                    if clean_table:
                        headers = clean_table[0]
                        table_md += "\n\n| " + " | ".join(headers) + " |\n"
                        table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                        for row in clean_table[1:]:
                            table_md += "| " + " | ".join(row) + " |\n"

            for line in text.split("\n"):
                line_str = line.strip()
                # Matches 'CHAPTER X', '1.1 Title', '2.3.2 Subtitle', but excludes simple numbered steps
                if re.match(r'^(?:CHAPTER\s+\d+|[0-9]+\.[0-9]+(?:\.[0-9]+)*\s+[A-Z][\w\s]+)', line_str):
                    if not re.search(r':\s*$', line_str) and len(line_str.split()) < 10:
                        current_heading = line_str

            page_content = text.strip() + (f"\n\n{table_md}" if table_md else "")
            if page_content.strip():
                elements.append(DocumentElement(
                    content=page_content.strip(),
                    metadata={
                        "source": os.path.basename(file_path),
                        "page_number": page_idx,
                        "section_heading": current_heading,
                        "doc_type": "pdf_native"
                    }
                ))
    return elements

def ingest_scanned_pdf_ocr(file_path: str, tesseract_cmd: Optional[str] = None) -> List[DocumentElement]:
    """Extract text from scanned PDF via OCR."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    elements = []
    if not os.path.exists(file_path):
        return elements

    doc = pymupdf.open(file_path)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_text = pytesseract.image_to_string(img).strip()
        first_line = ocr_text.split("\n")[0] if ocr_text else "Scanned Page"
        elements.append(DocumentElement(
            content=ocr_text,
            metadata={
                "source": os.path.basename(file_path),
                "page_number": page_idx + 1,
                "section_heading": first_line[:60],
                "doc_type": "pdf_scanned_ocr"
            }
        ))
    return elements

def ingest_docx(file_path: str) -> List[DocumentElement]:
    """Ingest structured DOCX document."""
    elements = []
    if not os.path.exists(file_path):
        return elements
    doc = Document(file_path)
    current_heading = "Preamble"
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading") or re.match(r'^[0-9]+(?:\.[0-9]+)*\s+[A-Z]', text):
            current_heading = text
        elements.append(DocumentElement(
            content=text,
            metadata={
                "source": os.path.basename(file_path),
                "page_number": 1,
                "section_heading": current_heading,
                "doc_type": "docx"
            }
        ))
    return elements

def ingest_txt(file_path: str) -> List[DocumentElement]:
    """Ingest TXT file with uppercase heading detection."""
    elements = []
    if not os.path.exists(file_path):
        return elements
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    buffer = []
    current_heading = "Header"
    for line in lines:
        if line.isupper() and len(line.strip()) > 3:
            if buffer:
                elements.append(DocumentElement(
                    content="".join(buffer).strip(),
                    metadata={
                        "source": os.path.basename(file_path),
                        "page_number": 1,
                        "section_heading": current_heading,
                        "doc_type": "txt"
                    }
                ))
                buffer = []
            current_heading = line.strip()
        buffer.append(line)
    if buffer:
        elements.append(DocumentElement(
            content="".join(buffer).strip(),
            metadata={
                "source": os.path.basename(file_path),
                "page_number": 1,
                "section_heading": current_heading,
                "doc_type": "txt"
            }
        ))
    return elements

def extract_images_and_figures(doc_path: str, output_img_dir: str = "outputs/images") -> List[Dict[str, Any]]:
    """Extracts embedded raster figures/diagrams to disk with source & page metadata."""
    os.makedirs(output_img_dir, exist_ok=True)
    if not os.path.exists(doc_path):
        return []
        
    doc = pymupdf.open(doc_path)
    image_metadata = []
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_filename = f"{os.path.basename(doc_path)}_p{page_idx+1}_img{img_idx}.{img_ext}"
            img_path = os.path.join(output_img_dir, img_filename)
            
            with open(img_path, "wb") as f:
                f.write(img_bytes)
                
            image_metadata.append({
                "image_path": img_path,
                "page_number": page_idx + 1,
                "source": os.path.basename(doc_path),
                "image_format": img_ext
            })
    return image_metadata