
import fitz


MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def validate_pdf_upload(uploaded_file):
    """
    Validate an uploaded PDF before document processing.

    Returns:
        (True, success_message) when valid.
        (False, error_message) when invalid.
    """

    if uploaded_file is None:
        return False, "No file was uploaded."

    filename = getattr(uploaded_file, "name", "")

    if not filename.lower().endswith(".pdf"):
        return False, "Only PDF files are supported."

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        return False, "The uploaded file is empty."

    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        return False, "The PDF exceeds the 10 MB upload limit."

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:

            if document.needs_pass:
                return False, "Password-protected PDFs are not supported."

            if document.page_count == 0:
                return False, "The PDF contains no pages."

    except Exception:
        return False, "The uploaded file is not a valid readable PDF."

    return True, "PDF validation successful."
