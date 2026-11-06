import unittest
from unittest.mock import patch

from document_chat import answer_uploaded_document


class TestDocumentChat(unittest.TestCase):

    def setUp(self):
        self.embedded_chunks = [
            {
                "id": "chunk_0",
                "text": "Payment processing information.",
                "page": 2,
                "embedding": [0.1] * 1536,
            },
            {
                "id": "chunk_1",
                "text": "Transaction status information.",
                "page": 5,
                "embedding": [0.2] * 1536,
            },
        ]

        self.retrieved_chunks = [
            {
                "id": "chunk_0",
                "text": "Payment processing information.",
                "page": 2,
                "embedding": [0.1] * 1536,
                "score": 0.95,
            }
        ]

    def test_rejects_empty_question(self):
        for question in ["", "   ", None]:
            with self.subTest(question=question):
                with self.assertRaisesRegex(
                    ValueError,
                    "The question cannot be empty."
                ):
                    answer_uploaded_document(
                        question,
                        self.embedded_chunks,
                    )

    def test_rejects_missing_uploaded_document(self):
        with self.assertRaisesRegex(
            ValueError,
            "No uploaded document is available for answering."
        ):
            answer_uploaded_document(
                "How are payments processed?",
                [],
            )

    @patch("document_chat.answer_with_chunks")
    @patch("document_chat.retrieve_document_chunks")
    def test_generates_answer_from_retrieved_chunks(
        self,
        mock_retrieve,
        mock_answer,
    ):
        expected_answer = "Payments are processed through the gateway [page 2]."

        mock_retrieve.return_value = self.retrieved_chunks
        mock_answer.return_value = (
            expected_answer,
            self.retrieved_chunks,
        )

        answer, sources = answer_uploaded_document(
            "How are payments processed?",
            self.embedded_chunks,
        )

        self.assertEqual(answer, expected_answer)
        self.assertEqual(sources, self.retrieved_chunks)

        mock_retrieve.assert_called_once_with(
            question="How are payments processed?",
            embedded_chunks=self.embedded_chunks,
            top_k=3,
        )

        mock_answer.assert_called_once_with(
            "How are payments processed?",
            self.retrieved_chunks,
        )

    @patch("document_chat.answer_with_chunks")
    @patch("document_chat.retrieve_document_chunks")
    def test_passes_custom_top_k_to_retrieval(
        self,
        mock_retrieve,
        mock_answer,
    ):
        mock_retrieve.return_value = self.retrieved_chunks
        mock_answer.return_value = (
            "Payment information [page 2].",
            self.retrieved_chunks,
        )

        answer_uploaded_document(
            "How are payments processed?",
            self.embedded_chunks,
            top_k=5,
        )

        mock_retrieve.assert_called_once_with(
            question="How are payments processed?",
            embedded_chunks=self.embedded_chunks,
            top_k=5,
        )

    @patch("document_chat.retrieve_document_chunks")
    def test_propagates_retrieval_failure(self, mock_retrieve):
        mock_retrieve.side_effect = RuntimeError(
            "Document retrieval unavailable"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Document retrieval unavailable"
        ):
            answer_uploaded_document(
                "How are payments processed?",
                self.embedded_chunks,
            )

    @patch("document_chat.answer_with_chunks")
    @patch("document_chat.retrieve_document_chunks")
    def test_propagates_answer_generation_failure(
        self,
        mock_retrieve,
        mock_answer,
    ):
        mock_retrieve.return_value = self.retrieved_chunks

        mock_answer.side_effect = RuntimeError(
            "Answer generation unavailable"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Answer generation unavailable"
        ):
            answer_uploaded_document(
                "How are payments processed?",
                self.embedded_chunks,
            )

        mock_answer.assert_called_once_with(
            "How are payments processed?",
            self.retrieved_chunks,
        )


if __name__ == "__main__":
    unittest.main()