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


if __name__ == "__main__":
    unittest.main()
