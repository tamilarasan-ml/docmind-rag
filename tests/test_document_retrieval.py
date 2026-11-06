
import unittest

from document_retrieval import retrieve_document_chunks
from types import ModuleType
from unittest.mock import Mock, patch


class TestDocumentRetrieval(unittest.TestCase):

    def setUp(self):
        self.query_vector = [1.0, 0.0, 0.0]

        self.chunks = [
            {
                "id": "chunk_0",
                "text": "First document passage.",
                "page": 1,
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "id": "chunk_1",
                "text": "Most relevant document passage.",
                "page": 3,
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "id": "chunk_2",
                "text": "Another document passage.",
                "page": 5,
                "embedding": [0.7, 0.7, 0.0],
            },
        ]

    def mock_embedding(self, text):
        return self.query_vector

    def test_ranks_chunks_by_cosine_similarity(self):
        results = retrieve_document_chunks(
            "What is the document about?",
            self.chunks,
            embedding_fn=self.mock_embedding,
        )

        self.assertEqual(results[0]["id"], "chunk_1")
        self.assertEqual(results[1]["id"], "chunk_2")
        self.assertEqual(results[2]["id"], "chunk_0")

    def test_returns_only_top_k_chunks(self):
        results = retrieve_document_chunks(
            "Find relevant information.",
            self.chunks,
            top_k=2,
            embedding_fn=self.mock_embedding,
        )

        self.assertEqual(len(results), 2)

    def test_preserves_original_page_numbers(self):
        results = retrieve_document_chunks(
            "Find relevant information.",
            self.chunks,
            top_k=1,
            embedding_fn=self.mock_embedding,
        )

        self.assertEqual(results[0]["page"], 3)

    def test_returns_similarity_score(self):
        results = retrieve_document_chunks(
            "Find relevant information.",
            self.chunks,
            embedding_fn=self.mock_embedding,
        )

        self.assertAlmostEqual(
            results[0]["similarity"],
            1.0,
        )

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            retrieve_document_chunks(
                "",
                self.chunks,
                embedding_fn=self.mock_embedding,
            )

    def test_rejects_empty_document_chunks(self):
        with self.assertRaises(ValueError):
            retrieve_document_chunks(
                "Find information.",
                [],
                embedding_fn=self.mock_embedding,
            )

    def test_rejects_zero_vector(self):
        with self.assertRaises(ValueError):
            retrieve_document_chunks(
                "Find information.",
                self.chunks,
                embedding_fn=lambda text: [0.0, 0.0, 0.0],
            )

    def test_rejects_mismatched_dimensions(self):
        invalid_chunks = [
            {
                "id": "chunk_invalid",
                "text": "Invalid vector.",
                "page": 1,
                "embedding": [1.0, 0.0],
            }
        ]

        with self.assertRaises(ValueError):
            retrieve_document_chunks(
                "Find information.",
                invalid_chunks,
                embedding_fn=self.mock_embedding,
            )

    def test_retrieval_is_scoped_to_supplied_chunks(self):
        document_a = [self.chunks[0]]
        document_b = [self.chunks[1]]

        results_a = retrieve_document_chunks(
            "Find information.",
            document_a,
            embedding_fn=self.mock_embedding,
        )

        results_b = retrieve_document_chunks(
            "Find information.",
            document_b,
            embedding_fn=self.mock_embedding,
        )

        self.assertEqual(
            [item["id"] for item in results_a],
            ["chunk_0"],
        )

        self.assertEqual(
            [item["id"] for item in results_b],
            ["chunk_1"],
        )

    def test_rejects_non_finite_query_embedding(self):
        invalid_query = [1.0, 0.0, 0.0]
        invalid_query[1] = float("nan")

        with self.assertRaisesRegex(
            ValueError,
            "Query embedding is invalid."
        ):
            retrieve_document_chunks(
                "Find information.",
                self.chunks,
                embedding_fn=lambda text: invalid_query,
            )

    def test_rejects_non_finite_document_embedding(self):
        invalid_chunks = [
            {
                "id": "chunk_invalid",
                "text": "Invalid document vector.",
                "page": 1,
                "embedding": [1.0, float("inf"), 0.0],
            }
        ]

        with self.assertRaisesRegex(
            ValueError,
            "A document embedding is invalid."
        ):
            retrieve_document_chunks(
                "Find information.",
                invalid_chunks,
                embedding_fn=self.mock_embedding,
            )

    def test_rejects_zero_document_embedding(self):
        invalid_chunks = [
            {
                "id": "chunk_invalid",
                "text": "Zero document vector.",
                "page": 1,
                "embedding": [0.0, 0.0, 0.0],
            }
        ]

        with self.assertRaisesRegex(
            ValueError,
            "A document embedding is invalid."
        ):
            retrieve_document_chunks(
                "Find information.",
                invalid_chunks,
                embedding_fn=self.mock_embedding,
            )

    def test_propagates_query_embedding_api_failure(self):
        def failing_embedding(text):
            raise RuntimeError("Query embedding API unavailable")

        with self.assertRaisesRegex(
            RuntimeError,
            "Query embedding API unavailable"
        ):
            retrieve_document_chunks(
                "Find information.",
                self.chunks,
                embedding_fn=failing_embedding,
            )

    def test_rejects_invalid_top_k(self):

        for invalid_top_k in [0, -1]:

            with self.subTest(top_k=invalid_top_k):
                with self.assertRaisesRegex(
                    ValueError,
                    "top_k must be at least 1."
                ):
                    retrieve_document_chunks(
                        "Find relevant information.",
                        self.chunks,
                        top_k=invalid_top_k,
                        embedding_fn=self.mock_embedding,
                    )

    def test_loads_default_embedding_function_lazily(self):
        mock_embed_module = ModuleType("embed")
        mock_embed_module.get_embedding = Mock(
            return_value=self.query_vector
        )

        with patch.dict(
            "sys.modules",
            {"embed": mock_embed_module}
        ):
            results = retrieve_document_chunks(
                "Find relevant information.",
                self.chunks,
            )

        mock_embed_module.get_embedding.assert_called_once_with(
            "Find relevant information."
        )
        self.assertEqual(len(results), 3)

    def test_rejects_invalid_query_embedding(self):
        invalid_embeddings = [
            [],
            [[1.0, 0.0, 0.0]],
        ]

        for invalid_embedding in invalid_embeddings:
            with self.subTest(embedding=invalid_embedding):
                with self.assertRaisesRegex(
                    ValueError,
                    "Invalid query embedding."
                ):
                    retrieve_document_chunks(
                        "Find relevant information.",
                        self.chunks,
                        embedding_fn=lambda text: invalid_embedding,
                    )

    def test_rejects_chunk_without_embedding(self):
        invalid_chunks = [
            {
                "id": "missing",
                "text": "Document chunk without an embedding.",
                "page": 1,
            }
        ]

        with self.assertRaisesRegex(
            ValueError,
            "A document chunk has no embedding."
        ):
            retrieve_document_chunks(
                "Find relevant information.",
                invalid_chunks,
                embedding_fn=self.mock_embedding,
            )

if __name__ == "__main__":
    unittest.main()
