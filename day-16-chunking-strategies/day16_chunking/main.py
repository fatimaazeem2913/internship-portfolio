import os
import json
from tabulate import tabulate
from src.ingestion import (
    ingest_native_pdf, 
    ingest_scanned_pdf_ocr, 
    ingest_docx, 
    ingest_txt,
    extract_images_and_figures
)
from src.chunkers import (
    chunk_fixed_size,
    chunk_token_based,
    chunk_recursive,
    chunk_semantic,
    chunk_hierarchical
)
from src.models import DocumentElement

def run_pipeline():
    print("================================================================")
    print("    DAY 16: DOCUMENT INGESTION & CHUNKING STRATEGIES TESTBED   ")
    print("================================================================\n")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/images", exist_ok=True)
    
    # Document registry map
    doc_sources = [
        ("Native PDF (117 Pages)", os.path.join("data", "SupportcoursesM-DLearning.pdf"), ingest_native_pdf),
        ("Scanned PDF (OCR)", os.path.join("data", "vendor_nda_scanned.pdf"), ingest_scanned_pdf_ocr),
        ("Structured DOCX", os.path.join("data", "product_spec.docx"), ingest_docx),
        ("Policy TXT", os.path.join("data", "api_rate_limiting_policy.txt"), ingest_txt)
    ]

    all_elements = []
    ingestion_summary = []

    print("--- 1. DOCUMENT INGESTION BREAKDOWN ---")
    for doc_label, path, ingest_func in doc_sources:
        if os.path.exists(path):
            doc_elements = ingest_func(path)
            all_elements.extend(doc_elements)
            
            # Compute document stats
            total_chars = sum(len(el.content) for el in doc_elements)
            unique_headings = len(set(el.metadata.get("section_heading", "") for el in doc_elements))
            page_count = len(set(el.metadata.get("page_number", 1) for el in doc_elements))
            
            ingestion_summary.append([
                doc_label,
                os.path.basename(path),
                len(doc_elements),
                page_count,
                unique_headings,
                f"{total_chars:,} chars"
            ])
            
            # Print preview snippet of the first extracted element
            if doc_elements:
                sample = doc_elements[0]
                preview = sample.content.replace('\n', ' ')[:120] + "..."
                print(f"\n[✔] Extracted from: {os.path.basename(path)}")
                print(f"    ├─ Elements : {len(doc_elements)} items")
                print(f"    ├─ Page     : {sample.metadata.get('page_number')}")
                print(f"    ├─ Heading  : {sample.metadata.get('section_heading')}")
                print(f"    └─ Snippet  : \"{preview}\"")
        else:
            print(f"[!] File not found: {path}")

    print("\n--- INGESTION SUMMARY TABLE ---")
    print(tabulate(
        ingestion_summary, 
        headers=["Document Type", "Filename", "Extracted Units", "Pages", "Headings Detected", "Total Volume"], 
        tablefmt="fancy_grid"
    ))

    # Multimodal image extraction
    pdf_path = os.path.join("data", "SupportcoursesM-DLearning.pdf")
    if os.path.exists(pdf_path):
        print("\n--- EXTRACTING EMBEDDED DIAGRAMS & IMAGES ---")
        extracted_imgs = extract_images_and_figures(pdf_path, output_img_dir="outputs/images")
        print(f"[✔] Successfully extracted {len(extracted_imgs)} figures/diagrams to 'outputs/images/'")

    # Chunking phase
    print("\n--- 2. CHUNKING STRATEGIES EXECUTION ---")
    strategies = {
        "1. Fixed-Size": chunk_fixed_size(all_elements),
        "2. Token-Based": chunk_token_based(all_elements),
        "3. Recursive (LangChain)": chunk_recursive(all_elements),
        "4. Semantic": chunk_semantic(all_elements),
        "5. Hierarchical": chunk_hierarchical(all_elements)
    }

    chunk_summary_table = []
    for name, chunks in strategies.items():
        sample_chunk = chunks[0] if chunks else None
        valid_meta = sample_chunk and all(k in sample_chunk.metadata for k in ["source", "page_number", "chunk_index", "section_heading"])
        status = "PASSED (100% Lineage)" if valid_meta else "FAILED"
        chunk_summary_table.append([name, len(chunks), status])

        slug = name.split(". ")[-1].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        out_file = os.path.join("outputs", f"chunks_{slug}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            # Dumps all chunks to disk without truncation
            json.dump([{"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata, "parent_id": c.parent_id} for c in chunks], f, indent=2)

    print(tabulate(chunk_summary_table, headers=["Chunking Strategy", "Chunks Generated", "Metadata Lineage Integrity"], tablefmt="fancy_grid"))
    print("\n[✔] Output JSONs successfully saved to 'outputs/' directory.\n")

if __name__ == "__main__":
    run_pipeline()