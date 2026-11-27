import os
import fitz  # PyMuPDF


def remove_repeated_boilerplate(pages, min_page_fraction=0.5):
    """
    Detects lines that repeat across many pages (headers/footers)
    and strips them out, since repeated content is rarely useful for search.
    """
    from collections import Counter

    line_counts = Counter()

    for page_data in pages:
        lines = set(
            line.strip()
            for line in page_data["text"].split("\n")
            if line.strip()
        )
        line_counts.update(lines)

    total_pages = len(pages)

    if total_pages == 0:
        return [], set()

    boilerplate_lines = {
        line
        for line, count in line_counts.items()
        if count / total_pages >= min_page_fraction
    }

    cleaned_pages = []

    for page_data in pages:
        lines = page_data["text"].split("\n")
        kept_lines = [
            line for line in lines
            if line.strip() not in boilerplate_lines
        ]

        cleaned_pages.append({
            "page": page_data["page"],
            "text": "\n".join(kept_lines)
        })

    return cleaned_pages, boilerplate_lines


def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF and extracts all text, page by page.

    Returns:
        [{"page": 1, "text": "..."}, ...]
    """
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()

            if text.strip():
                pages.append({
                    "page": page_num,
                    "text": text
                })

    return pages


def is_low_quality_chunk(text):
    """
    Rejects empty content and chunks dominated by numbers or symbols.

    Short but meaningful text is retained, including unique identifiers
    and marker-only pages.
    """
    words = text.split()

    if not words:
        return True

    alpha_words = [
        word
        for word in words
        if any(char.isalpha() for char in word)
    ]

    return len(alpha_words) / len(words) < 0.6


def chunk_text(pages, chunk_size=500, overlap=50, max_chunks=None):
    """
    Splits page text into overlapping word chunks.

    Short meaningful content is preserved.

    max_chunks:
        Optional maximum number of chunks.
        None preserves unlimited chunking.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    if max_chunks is not None and max_chunks < 1:
        raise ValueError("max_chunks must be at least 1.")

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

                if max_chunks is not None and len(chunks) >= max_chunks:
                    raise ValueError(
                        f"The document exceeds the {max_chunks}-chunk limit."
                    )

                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text_str,
                    "page": page_num
                })

                chunk_id += 1

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    # Change this to your actual PDF filename.
    pdf_filename = "JQL.pdf"
    pdf_path = os.path.join("data", pdf_filename)

    print(f"Reading: {pdf_path}")

    pages = extract_text_from_pdf(pdf_path)

    print(f"Extracted {len(pages)} pages with text.")

    # Remove repeated headers and footers.
    pages, boilerplate = remove_repeated_boilerplate(pages)

    print(f"Removed {len(boilerplate)} repeated boilerplate lines.")

    chunks = chunk_text(pages)

    print(f"Created {len(chunks)} chunks.\n")

    if chunks:
        print("--- Sample chunk ---")
        print(f"ID: {chunks[0]['id']}")
        print(f"Page: {chunks[0]['page']}")
        print(f"Text preview: {chunks[0]['text'][:300]}...")

        print("\n--- Full text of chunk_0 ---")
        print(chunks[0]["text"])

    import json

    output_path = os.path.join("data", "chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nSaved {len(chunks)} chunks to {output_path}")
