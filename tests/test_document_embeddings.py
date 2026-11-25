
import unittest

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from document_embeddings import embed_document_chunks


def make_batch_response(embeddings):
    """Create mock embedding response records."""
    return [
        SimpleNamespace(
            index=index,
            embedding=embedding,
        )
        for index, embedding in embeddings
    ]


class TestDocumentEmbeddings(unittest.TestCase):

    def setUp(self):
        self.fake_embedding = [0.1] * 1536

    def mock_embedding(self, text):
        return self.fake_embedding

    # --------------------------------------------------
    # Batch embedding tests
    # --------------------------------------------------

    def test_uses_five_batches_for_100_chunks(self):
        chunks = [
            {
                "id": f"chunk_{i}",
                "text": f"Document content for chunk {i}.",
                "page": 1,
            }
            for i in range(100)
        ]

        mock_embed_module = ModuleType("embed")
        mock_embed_module.get_embeddings = Mock(
            side_effect=lambda texts: make_batch_response(
                [
                    (i, self.fake_embedding)
                    for i in range(len(texts))
                ]
            )
        )

        with patch.dict(
            "sys.modules",
            {"embed": mock_embed_module},
        ):
            result = embed_document_chunks(chunks)

        self.assertEqual(len(result), 100)

        self.assertEqual(
            mock_embed_module.get_embeddings.call_count,
            5,
        )

        batch_sizes = [
            len(call.args[0])
            for call in mock_embed_module.get_embeddings.call_args_list
        ]

        self.assertEqual(
            batch_sizes,
            [20, 20, 20, 20, 20],
        )

    def test_handles_final_partial_batch(self):
        chunks = [
            {
                "id": f"chunk_{i}",
                "text": f"Document content for chunk {i}.",
                "page": 1,
            }
            for i in range(45)
        ]

        mock_embed_module = ModuleType("embed")
        mock_embed_module.get_embeddings = Mock(
            side_effect=lambda texts: make_batch_response(
                [
                    (i, self.fake_embedding)
                    for i in range(len(texts))
                ]
            )
        )

        with patch.dict(
            "sys.modules",
            {"embed": mock_embed_module},
        ):
            result = embed_document_chunks(chunks)

        self.assertEqual(len(result), 45)

        batch_sizes = [
            len(call.args[0])
            for call in mock_embed_module.get_embeddings.call_args_list
        ]

        self.assertEqual(
            batch_sizes,
            [20, 20, 5],
        )

    def test_maps_batch_response_indexes_to_original_chunks(self):
        chunks = [
            {
                "id": "chunk_0",
                "text": "First chunk.",
                "page": 1,
            },
            {
                "id": "chunk_1",
                "text": "Second chunk.",
                "page": 2,
            },
        ]

        first_embedding = [0.2] * 1536
        second_embedding = [0.3] * 1536

        mock_embed_module = ModuleType("embed")
        mock_embed_module.get_embeddings = Mock(
            return_value=make_batch_response([
                (1, second_embedding),
                (0, first_embedding),
            ])
        )

        with patch.dict(
            "sys.modules",
            {"embed": mock_embed_module},
        ):
            result = embed_document_chunks(chunks)

        self.assertEqual(
            result[0]["embedding"],
            first_embedding,
        )

        self.assertEqual(
            result[1]["embedding"],
            second_embedding,
        )

        self.assertEqual(result[0]["page"], 1)
        self.assertEqual(result[1]["page"], 2)


    def test_rejects_missing_batch_response_index(self):
        chunks = [
            {"id": "chunk_0", "text": "First chunk.", "page": 1},
            {"id": "chunk_1", "text": "Second chunk.", "page": 2},
        ]

        mock_embed_module = ModuleType("embed")
        mock_embed_module.get_embeddings = Mock(
            return_value=make_batch_response([
                (0, self.fake_embedding),
            ])
        )

        with patch.dict("sys.modules", {"embed": mock_embed_module}):
            with self.assertRaisesRegex(
                ValueError,
                "unexpected number of vectors",
            ):
                embed_document_chunks(chunks)

    def test_rejects_duplicate_batch_response_indexes(self):
        chunks = [
            {"id": "chunk_0", "text": "First chunk.", "page": 1},
            {"id": "chunk_1", "text": "Second chunk.", "page": 2},
        ]

        mock_embed_module = ModuleType("embed")
        mock_embed_module.get_embeddings = Mock(
            return_value=make_batch_response([
                (0, self.fake_embedding),
                (0, self.fake_embedding),
            ])
        )

        with patch.dict("sys.modules", {"embed": mock_embed_module}):
            with self.assertRaisesRegex(
                ValueError,
                "duplicate response indexes",
            ):
                embed_document_chunks(chunks)

    def test_rejects_unexpected_batch_response_count(self):
        chunks = [
            {"id": "chunk_0", "text": "First chunk.", "page": 1},
            {"id": "chunk_1", "text": "Second chunk.", "page": 2},
        ]

        mock_embed_module = ModuleType("embed")
        mock_embed_module.get_embeddings = Mock(
            return_value=make_batch_response([
                (0, self.fake_embedding),
            ])
        )

        with patch.dict("sys.modules", {"embed": mock_embed_module}):
            with self.assertRaisesRegex(
                ValueError,
                "unexpected number of vectors",
            ):
                embed_document_chunks(chunks)

    def test_propagates_batch_embedding_api_failure(self):
        chunks = [
            {"id": "chunk_0", "text": "First chunk.", "page": 1},
        ]

        mock_embed_module = ModuleType("embed")
        mock_embed_module.get_embeddings = Mock(
            side_effect=RuntimeError("Azure batch API unavailable")
        )

        with patch.dict("sys.modules", {"embed": mock_embed_module}):
            with self.assertRaisesRegex(
                RuntimeError,
                "Azure batch API unavailable",
            ):
                embed_document_chunks(chunks)


    # --------------------------------------------------
    # Existing embedding tests
    # --------------------------------------------------

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

    def test_rejects_scalar_embedding(self):
        with self.assertRaisesRegex(
            ValueError,
            "Embedding contains invalid numerical values.",
        ):
            embed_document_chunks(
                [{"text": "Test document chunk"}],
                embedding_fn=lambda text: 0.5,
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

        for invalid_value in [
            float("inf"),
            float("-inf"),
        ]:
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
            "Embedding API unavailable",
        ):
            embed_document_chunks(
                chunks,
                embedding_fn=failing_embedding,
            )

    def test_loads_default_embedding_function_lazily(self):
        mock_embed_module = ModuleType("embed")

        mock_embed_module.get_embeddings = Mock(
            return_value=make_batch_response([
                (0, self.fake_embedding),
            ])
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

        mock_embed_module.get_embeddings.assert_called_once_with(
            ["Default embedding function test."]
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
            "Embedding contains invalid numerical values.",
        ):
            embed_document_chunks(
                chunks,
                embedding_fn=lambda text: invalid_embedding,
            )

    def test_accepts_exactly_100_embedding_chunks(self):
        chunks = [
            {
                "id": f"chunk_{i}",
                "text": f"Meaningful document content for chunk {i}.",
                "page": 1,
            }
            for i in range(100)
        ]

        mock_embedding = Mock(
            return_value=self.fake_embedding,
        )

        result = embed_document_chunks(
            chunks,
            embedding_fn=mock_embedding,
        )

        self.assertEqual(len(result), 100)
        self.assertEqual(mock_embedding.call_count, 100)

    def test_rejects_more_than_100_chunks_before_embedding(self):
        chunks = [
            {
                "id": f"chunk_{i}",
                "text": f"Meaningful document content for chunk {i}.",
                "page": 1,
            }
            for i in range(101)
        ]

        mock_embedding = Mock(
            return_value=self.fake_embedding,
        )

        with self.assertRaisesRegex(
            ValueError,
            "exceeds the 100-chunk embedding limit",
        ):
            embed_document_chunks(
                chunks,
                embedding_fn=mock_embedding,
            )

        mock_embedding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
