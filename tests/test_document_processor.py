
import unittest
import fitz

from document_processor import process_uploaded_pdf
from unittest.mock import patch


def create_test_pdf(page_texts):
    """Create an in-memory PDF with wrapped text."""
    document = fitz.open()

    for text in page_texts:
        page = document.new_page()

        if text.strip():
            page.insert_textbox(
                fitz.Rect(50, 50, 550, 750),
                text,
                fontsize=10,
            )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


class TestDocumentProcessor(unittest.TestCase):

    def test_processes_valid_pdf(self):
        text = (
            "DOCMIND document processing test. "
            "This sample contains enough meaningful words "
            "to create a valid text chunk for retrieval. "
        ) * 5

        chunks = process_uploaded_pdf(
            create_test_pdf([text])
        )

        self.assertTrue(len(chunks) > 0)
        self.assertIn("text", chunks[0])
        self.assertIn("page", chunks[0])
        self.assertIn("id", chunks[0])
        self.assertEqual(chunks[0]["page"], 1)

    def test_preserves_original_page_numbers(self):
        page_one = (
            "First page contains meaningful document content. "
        ) * 5

        page_three = (
            "Third page contains additional meaningful content. "
        ) * 5

        pdf_bytes = create_test_pdf([
            page_one,
            "",
            page_three,
        ])

        chunks = process_uploaded_pdf(pdf_bytes)

        page_numbers = {chunk["page"] for chunk in chunks}

        self.assertIn(1, page_numbers)
        self.assertIn(3, page_numbers)
        self.assertNotIn(2, page_numbers)

    def test_rejects_empty_bytes(self):
        with self.assertRaises(ValueError):
            process_uploaded_pdf(b"")

    def test_rejects_invalid_pdf(self):
        with self.assertRaises(ValueError):
            process_uploaded_pdf(b"not a valid PDF")

    def test_rejects_pdf_without_extractable_text(self):
        pdf_bytes = create_test_pdf([""])

        with self.assertRaises(ValueError):
            process_uploaded_pdf(pdf_bytes)

    def test_rejects_password_protected_pdf(self):
        document = fitz.open()
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Protected document test"
        )

        pdf_bytes = document.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )

        document.close()

        with self.assertRaisesRegex(
            ValueError,
            "Password-protected PDFs are not supported."
        ):
            process_uploaded_pdf(pdf_bytes)

    def test_rejects_scanned_pdf_without_ocr(self):
        document = fitz.open()
        page = document.new_page()

        pixmap = fitz.Pixmap(
            fitz.csRGB,
            fitz.IRect(0, 0, 200, 100),
            0,
        )
        pixmap.clear_with(255)

        image_bytes = pixmap.tobytes("png")

        page.insert_image(
            fitz.Rect(50, 50, 250, 150),
            stream=image_bytes,
        )

        pdf_bytes = document.tobytes()
        document.close()

        with self.assertRaisesRegex(
            ValueError,
            "Scanned PDFs are not supported yet."
        ):
            process_uploaded_pdf(pdf_bytes)

    def test_processes_mixed_text_and_image_pdf(self):
        document = fitz.open()

        # Page 1: Searchable text
        page_one = document.new_page()
        page_one.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            "Mixed PDF test with searchable text. " * 10,
            fontsize=10,
        )

        # Page 2: Image only
        page_two = document.new_page()

        pixmap = fitz.Pixmap(
            fitz.csRGB,
            fitz.IRect(0, 0, 200, 100),
            0,
        )
        pixmap.clear_with(255)

        page_two.insert_image(
            fitz.Rect(50, 50, 250, 150),
            stream=pixmap.tobytes("png"),
        )

        # Page 3: Searchable text
        page_three = document.new_page()
        page_three.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            "Final page contains searchable document content. " * 10,
            fontsize=10,
        )

        pdf_bytes = document.tobytes()
        document.close()

        chunks = process_uploaded_pdf(pdf_bytes)

        page_numbers = {chunk["page"] for chunk in chunks}

        self.assertIn(1, page_numbers)
        self.assertIn(3, page_numbers)
        self.assertNotIn(2, page_numbers)

    @patch("document_processor.chunk_text", return_value=[])
    def test_rejects_pdf_when_no_chunks_are_created(self, mock_chunk_text):
        text = (
            "This PDF contains valid extractable text, "
            "but the chunking function returns no chunks."
        )

        pdf_bytes = create_test_pdf([text])

        with self.assertRaisesRegex(
            ValueError,
            "No usable text chunks could be created from the PDF."
        ):
            process_uploaded_pdf(pdf_bytes)

        mock_chunk_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
