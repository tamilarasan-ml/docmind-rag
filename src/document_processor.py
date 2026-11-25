import fitz

from ingest import chunk_text
from document_upload import MAX_PDF_PAGES


MAX_EXTRACTED_TEXT_CHARS = 500_000


def process_uploaded_pdf(pdf_bytes):
    """
    Extract and chunk an uploaded PDF entirely in memory.

    Preserves original page numbers and source text.
    Does not apply repeated-boilerplate removal.
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:

            if document.needs_pass:
                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            if document.page_count > MAX_PDF_PAGES:
                raise ValueError(
                    f"The PDF exceeds the {MAX_PDF_PAGES}-page limit."
                )

            pages = []
            total_extracted_chars = 0

            for page_number, page in enumerate(document, start=1):
                text = page.get_text()
                total_extracted_chars += len(text)

                if total_extracted_chars > MAX_EXTRACTED_TEXT_CHARS:
                    raise ValueError(
                        "The PDF exceeds the 500,000-character extracted-text limit."
                    )

                if text.strip():
                    pages.append({
                        "page": page_number,
                        "text": text,
                    })

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Unable to process the uploaded PDF."
        ) from exc

    if not pages:
        raise ValueError(
            "The PDF contains no extractable text. "
            "Scanned PDFs are not supported yet."
        )

    chunks = chunk_text(
        pages,
        chunk_size=500,
        overlap=50,
        max_chunks=100,
    )

    if not chunks:
        raise ValueError(
            "No usable text chunks could be created from the PDF."
        )

    return chunks