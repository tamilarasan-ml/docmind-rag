
import unittest
import fitz

from document_processor import process_uploaded_pdf
from unittest.mock import MagicMock, patch


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

    def test_accepts_pdf_at_maximum_page_limit(self):
        from document_upload import MAX_PDF_PAGES

        mock_pages = [MagicMock() for _ in range(MAX_PDF_PAGES)]
        for mock_page in mock_pages:
            mock_page.get_text.return_value = (
                "Meaningful document content. " * 20
            )

        mock_document = MagicMock()
        mock_document.__enter__.return_value = mock_document
        mock_document.needs_pass = False
        mock_document.page_count = MAX_PDF_PAGES
        mock_document.__iter__.return_value = iter(mock_pages)

        expected_chunks = [
            {"id": "chunk_0", "text": "valid", "page": 1}
        ]

        with patch(
            "document_processor.fitz.open",
            return_value=mock_document,
        ):
            with patch(
                "document_processor.chunk_text",
                return_value=expected_chunks,
            ):
                chunks = process_uploaded_pdf(b"mock-pdf")

        self.assertEqual(chunks, expected_chunks)
        self.assertEqual(
            sum(page.get_text.call_count for page in mock_pages),
            MAX_PDF_PAGES,
        )

    def test_rejects_pdf_exceeding_maximum_page_limit_before_extraction(self):
        from document_upload import MAX_PDF_PAGES

        mock_page = MagicMock()

        mock_document = MagicMock()
        mock_document.__enter__.return_value = mock_document
        mock_document.needs_pass = False
        mock_document.page_count = MAX_PDF_PAGES + 1
        mock_document.__iter__.return_value = iter([mock_page])

        with patch(
            "document_processor.fitz.open",
            return_value=mock_document,
        ):
            with self.assertRaisesRegex(
                ValueError,
                f"The PDF exceeds the {MAX_PDF_PAGES}-page limit.",
            ):
                process_uploaded_pdf(b"mock-pdf")

        mock_document.__iter__.assert_not_called()
        mock_page.get_text.assert_not_called()

    def test_accepts_text_at_maximum_character_limit(self):
        from document_processor import MAX_EXTRACTED_TEXT_CHARS

        page_text = "A" * MAX_EXTRACTED_TEXT_CHARS
        mock_page = type("MockPage", (), {
            "get_text": lambda self: page_text
        })()

        mock_document = type("MockDocument", (), {
            "needs_pass": False,
            "page_count": 1,
            "__iter__": lambda self: iter([mock_page]),
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
        })()

        expected_chunks = [{"id": "chunk_0", "text": "valid", "page": 1}]

        with patch("document_processor.fitz.open", return_value=mock_document):
            with patch(
                "document_processor.chunk_text",
                return_value=expected_chunks,
            ) as mock_chunk:
                chunks = process_uploaded_pdf(b"mock-pdf")

        self.assertEqual(chunks, expected_chunks)
        mock_chunk.assert_called_once()

    def test_rejects_text_exceeding_maximum_character_limit(self):
        from document_processor import MAX_EXTRACTED_TEXT_CHARS

        page_text = "A" * (MAX_EXTRACTED_TEXT_CHARS + 1)
        mock_page = type("MockPage", (), {
            "get_text": lambda self: page_text
        })()

        mock_document = type("MockDocument", (), {
            "needs_pass": False,
            "page_count": 1,
            "__iter__": lambda self: iter([mock_page]),
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
        })()

        with patch("document_processor.fitz.open", return_value=mock_document):
            with patch("document_processor.chunk_text") as mock_chunk:
                with self.assertRaisesRegex(
                    ValueError,
                    "500,000-character extracted-text limit",
                ):
                    process_uploaded_pdf(b"mock-pdf")

        mock_chunk.assert_not_called()

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
