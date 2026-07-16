import os
import fitz  # this is PyMuPDF's import name — confusing but correct


def remove_repeated_boilerplate(pages, min_page_fraction=0.5):
    """
    Detects lines that repeat across many pages (headers/footers)
    and strips them out, since repeated content is rarely useful for search.
    """
    from collections import Counter

    line_counts = Counter()
    for page_data in pages:
        lines = set(line.strip() for line in page_data["text"].split("\n") if line.strip())
        line_counts.update(lines)

    total_pages = len(pages)
    boilerplate_lines = {
        line for line, count in line_counts.items()
        if count / total_pages >= min_page_fraction
    }

    cleaned_pages = []
    for page_data in pages:
        lines = page_data["text"].split("\n")
        kept_lines = [l for l in lines if l.strip() not in boilerplate_lines]
        cleaned_pages.append({
            "page": page_data["page"],
            "text": "\n".join(kept_lines)
        })

    return cleaned_pages, boilerplate_lines
    
def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF and extracts all text, page by page.
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():  # skip blank pages
            pages.append({"page": page_num, "text": text})
    
    doc.close()
    return pages


def chunk_text(pages, chunk_size=500, overlap=50):
    """
    Splits page text into overlapping word chunks.
    chunk_size = target words per chunk
    overlap = words repeated between consecutive chunks (preserves context across boundaries)
    """
    chunks = []
    chunk_id = 0

    for page_data in pages:
        words = page_data["text"].split()
        page_num = page_data["page"]

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text_str = " ".join(chunk_words)

            if not is_low_quality_chunk(chunk_text_str):
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text_str,
                    "page": page_num
                })
                chunk_id += 1

            start += chunk_size - overlap  # move forward, but overlap with previous chunk

    return chunks
def is_low_quality_chunk(text, min_words=20):
    """
    Filters out chunks that are likely noise: too short, 
    or mostly numbers/symbols rather than real sentences.
    """
    words = text.split()
    if len(words) < min_words:
        return True
    
    # Count how many "words" are mostly digits/symbols (phone numbers, addresses, etc.)
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if len(alpha_words) / len(words) < 0.6:  # less than 60% real text
        return True
    
    return False

if __name__ == "__main__":
    # CHANGE THIS to your actual PDF filename
    pdf_filename = "JQL.pdf"
    pdf_path = os.path.join("data", pdf_filename)

    print(f"Reading: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages with text.")

    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages with text.")

    # Skip the cover/contact page (page 1) — it's boilerplate, not real content
    pages, boilerplate = remove_repeated_boilerplate(pages)
    print(f"Removed {len(boilerplate)} repeated boilerplate lines.")

    chunks = chunk_text(pages)

    chunks = chunk_text(pages)
    print(f"Created {len(chunks)} chunks.\n")

    # Show the first chunk so we can eyeball the quality
    print("--- Sample chunk ---")
    print(f"ID: {chunks[0]['id']}")
    print(f"Page: {chunks[0]['page']}")
    print(f"Text preview: {chunks[0]['text'][:300]}...")

    print("\n--- Full text of chunk_0 (to verify filter logic) ---")
    print(chunks[0]['text'])

    import json
    output_path = os.path.join("data", "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(chunks)} chunks to {output_path}")