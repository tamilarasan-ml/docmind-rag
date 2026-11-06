
import unittest
import io
import fitz

from document_upload import validate_pdf_upload


class DocumentUploadTests(unittest.TestCase):

    def create_valid_pdf(self):
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "DOCMIND test document")

        pdf_bytes = document.tobytes()
        document.close()

        return pdf_bytes

    def test_accepts_valid_pdf(self):
        uploaded_file = io.BytesIO(self.create_valid_pdf())
        uploaded_file.name = "sample.pdf"

        valid, message = validate_pdf_upload(uploaded_file)

        self.assertTrue(valid)
        self.assertEqual(message, "PDF validation successful.")

    def test_rejects_missing_file(self):
        valid, message = validate_pdf_upload(None)

        self.assertFalse(valid)
        self.assertEqual(message, "No file was uploaded.")

    def test_rejects_non_pdf_extension(self):
        uploaded_file = io.BytesIO(b"sample content")
        uploaded_file.name = "sample.txt"

        valid, message = validate_pdf_upload(uploaded_file)

        self.assertFalse(valid)
        self.assertEqual(message, "Only PDF files are supported.")

    def test_rejects_empty_file(self):
        uploaded_file = io.BytesIO(b"")
        uploaded_file.name = "empty.pdf"

        valid, message = validate_pdf_upload(uploaded_file)

        self.assertFalse(valid)
        self.assertEqual(message, "The uploaded file is empty.")

    def test_rejects_oversized_file(self):
        from document_upload import MAX_UPLOAD_SIZE_BYTES

        uploaded_file = io.BytesIO(
            b"x" * (MAX_UPLOAD_SIZE_BYTES + 1)
        )
        uploaded_file.name = "large.pdf"

        valid, message = validate_pdf_upload(uploaded_file)

        self.assertFalse(valid)
        self.assertEqual(
            message,
            "The PDF exceeds the 10 MB upload limit."
        )

    def test_rejects_password_protected_pdf(self):
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Protected DOCMIND test")

        pdf_bytes = document.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )
        document.close()

        uploaded_file = io.BytesIO(pdf_bytes)
        uploaded_file.name = "protected.pdf"

        valid, message = validate_pdf_upload(uploaded_file)

        self.assertFalse(valid)
        self.assertEqual(
            message,
            "Password-protected PDFs are not supported."
        )

    def test_rejects_malformed_pdf(self):
        uploaded_file = io.BytesIO(b"This is not a PDF")
        uploaded_file.name = "malformed.pdf"

        valid, message = validate_pdf_upload(uploaded_file)

        self.assertFalse(valid)
        self.assertEqual(
            message,
            "The uploaded file is not a valid readable PDF."
        )

    def test_rejects_zero_page_pdf(self):
        from unittest.mock import patch, MagicMock

        uploaded_file = io.BytesIO(b"mock-pdf-content")
        uploaded_file.name = "zero_page.pdf"

        mock_document = MagicMock()
        mock_document.needs_pass = False
        mock_document.page_count = 0

        mock_document.__enter__.return_value = mock_document

        with patch(
            "document_upload.fitz.open",
            return_value=mock_document,
        ):
            valid, message = validate_pdf_upload(uploaded_file)

        self.assertFalse(valid)
        self.assertEqual(
            message,
            "The PDF contains no pages."
        )

        mock_document.__enter__.assert_called_once()
        mock_document.__exit__.assert_called_once()

if __name__ == "__main__":
    unittest.main()
