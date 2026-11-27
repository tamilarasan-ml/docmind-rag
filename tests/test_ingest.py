import unittest

from ingest import chunk_text


class TestChunkTextLimits(unittest.TestCase):

    def setUp(self):
        self.pages = [{
            "page": 1,
            "text": "content " * 45020
        }]

    def test_accepts_exactly_100_chunks(self):
        pages = [{
            "page": 1,
            "text": "content " * 45000
        }]

        chunks = chunk_text(pages, max_chunks=100)

        self.assertEqual(len(chunks), 100)

    def test_rejects_chunk_101(self):
        with self.assertRaisesRegex(
            ValueError,
            "exceeds the 100-chunk limit"
        ):
            chunk_text(self.pages, max_chunks=100)

    def test_default_preserves_unlimited_chunking(self):
        chunks = chunk_text(self.pages)

        self.assertEqual(len(chunks), 101)


class TestShortContentPreservation(unittest.TestCase):

    def test_preserves_short_marker_content(self):
        pages = [{
            "page": 41,
            "text": "DOCMIND-ALPHA-731"
        }]

        chunks = chunk_text(pages)

        self.assertEqual(len(chunks), 1)
        self.assertIn(
            "DOCMIND-ALPHA-731",
            chunks[0]["text"]
        )
        self.assertEqual(chunks[0]["page"], 41)

    def test_preserves_short_meaningful_content(self):
        pages = [{
            "page": 2,
            "text": "This document contains important information."
        }]

        chunks = chunk_text(pages)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page"], 2)

    def test_rejects_empty_content(self):
        pages = [{
            "page": 3,
            "text": ""
        }]

        chunks = chunk_text(pages)

        self.assertEqual(len(chunks), 0)

    def test_rejects_numeric_and_symbol_only_content(self):
        pages = [{
            "page": 4,
            "text": "12345 67890 !!! ### $$$"
        }]

        chunks = chunk_text(pages)

        self.assertEqual(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
