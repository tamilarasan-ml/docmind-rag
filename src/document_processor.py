import fitz

from ingest import chunk_text


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

            pages = []

            for page_number, page in enumerate(document, start=1):
                text = page.get_text()

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
    )

    if not chunks:
        raise ValueError(
            "No usable text chunks could be created from the PDF."
        )

    return chunks