
import unittest

from document_embeddings import embed_document_chunks
from types import ModuleType
from unittest.mock import Mock, patch


class TestDocumentEmbeddings(unittest.TestCase):

    def setUp(self):
        self.fake_embedding = [0.1] * 1536

    def mock_embedding(self, text):
        return self.fake_embedding

    def test_generates_embeddings_for_chunks(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "First meaningful document chunk.",
                "page": 1,
            },
            {
                "id": "chunk_1",
                "text": "Second meaningful document chunk.",
                "page": 2,
            },
        ]

        result = embed_document_chunks(
            chunks,
            embedding_fn=self.mock_embedding,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            len(result[0]["embedding"]),
            1536,
        )
        self.assertEqual(result[0]["page"], 1)
        self.assertEqual(result[1]["page"], 2)

    def test_preserves_original_chunks(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "Original document content.",
                "page": 1,
            }
        ]

        embed_document_chunks(
            chunks,
            embedding_fn=self.mock_embedding,
        )

        self.assertNotIn("embedding", chunks[0])

    def test_rejects_empty_chunk_list(self):
        with self.assertRaises(ValueError):
            embed_document_chunks(
                [],
                embedding_fn=self.mock_embedding,
            )

    def test_rejects_empty_chunk_text(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "   ",
                "page": 1,
            }
        ]

        with self.assertRaises(ValueError):
            embed_document_chunks(
                chunks,
                embedding_fn=self.mock_embedding,
            )

    def test_rejects_empty_embedding(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "Document content.",
                "page": 1,
            }
        ]

        with self.assertRaises(ValueError):
            embed_document_chunks(
                chunks,
                embedding_fn=lambda text: [],
            )

    def test_rejects_unexpected_embedding_dimensions(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "Document content.",
                "page": 1,
            }
        ]

        with self.assertRaises(ValueError):
            embed_document_chunks(
                chunks,
                embedding_fn=lambda text: [0.1] * 100,
            )

    def test_rejects_non_finite_embedding_values(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "Document content.",
                "page": 1,
            }
        ]

        invalid_embedding = [0.1] * 1536
        invalid_embedding[10] = float("nan")

        with self.assertRaises(ValueError):
            embed_document_chunks(
                chunks,
                embedding_fn=lambda text: invalid_embedding,
            )

    def test_rejects_infinite_embedding_values(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "Document content.",
                "page": 1,
            }
        ]

        for invalid_value in [float("inf"), float("-inf")]:
            invalid_embedding = [0.1] * 1536
            invalid_embedding[10] = invalid_value

            with self.assertRaises(ValueError):
                embed_document_chunks(
                    chunks,
                    embedding_fn=lambda text: invalid_embedding,
                )

    def test_rejects_non_numeric_embedding_values(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "Document content.",
                "page": 1,
            }
        ]

        invalid_embedding = [0.1] * 1536
        invalid_embedding[10] = "invalid"

        with self.assertRaises(ValueError):
            embed_document_chunks(
                chunks,
                embedding_fn=lambda text: invalid_embedding,
            )

    def test_rejects_zero_embedding_vector(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "Document content.",
                "page": 1,
            }
        ]

        zero_embedding = [0.0] * 1536

        with self.assertRaisesRegex(
            ValueError,
            "Embedding vector cannot contain only zeros."
        ):
            embed_document_chunks(
                chunks,
                embedding_fn=lambda text: zero_embedding,
            )

    def test_propagates_embedding_api_failure(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "Document content.",
                "page": 1,
            }
        ]

        def failing_embedding(text):
            raise RuntimeError("Embedding API unavailable")

        with self.assertRaisesRegex(
            RuntimeError,
            "Embedding API unavailable"
        ):
            embed_document_chunks(
                chunks,
                embedding_fn=failing_embedding,
            )

    def test_loads_default_embedding_function_lazily(self):

        mock_embed_module = ModuleType("embed")

        mock_embed_module.get_embedding = Mock(
            return_value=self.fake_embedding
        )

        chunks = [
            {
                "id": "chunk_0",
                "text": "Default embedding function test.",
                "page": 1,
            }
        ]

        with patch.dict(
            "sys.modules",
            {"embed": mock_embed_module},
        ):
            result = embed_document_chunks(chunks)

        mock_embed_module.get_embedding.assert_called_once_with(
            "Default embedding function test."
        )

        self.assertEqual(len(result), 1)

        self.assertEqual(
            result[0]["embedding"],
            self.fake_embedding,
        )

    def test_rejects_embedding_when_numeric_validation_raises_overflow(self):

        chunks = [
            {
                "id": "chunk_0",
                "text": "Numerical overflow validation test.",
                "page": 1,
            }
        ]

        invalid_embedding = [10**10000] + [0.1] * 1535

        with self.assertRaisesRegex(
            ValueError,
            "Embedding contains invalid numerical values."
        ):
            embed_document_chunks(
                chunks,
                embedding_fn=lambda text: invalid_embedding,
            )

if __name__ == "__main__":
    unittest.main()
