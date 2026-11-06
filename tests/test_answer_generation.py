import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from answer_generation import (
    answer_with_chunks,
    build_context,
    fingerprint,
)


class TestAnswerGeneration(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {
                "id": "chunk_0",
                "text": "Payments are processed through the payment gateway.",
                "page": 2,
            },
            {
                "id": "chunk_1",
                "text": "Transaction status can be verified using the reference ID.",
                "page": 5,
            },
        ]

    def test_builds_context_with_original_page_references(self):
        context = build_context(self.chunks)

        self.assertIn("[Source: page 2]", context)
        self.assertIn("[Source: page 5]", context)
        self.assertIn(self.chunks[0]["text"], context)
        self.assertIn(self.chunks[1]["text"], context)
        self.assertIn("\n\n---\n\n", context)

    def test_generates_deterministic_fingerprint(self):
        first = fingerprint("DOCMIND")
        second = fingerprint("DOCMIND")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)

    def test_rejects_empty_question(self):
        for question in ["", "   ", None]:
            with self.subTest(question=question):
                with self.assertRaisesRegex(
                    ValueError,
                    "The question cannot be empty."
                ):
                    answer_with_chunks(question, self.chunks)

    def test_returns_insufficient_information_for_empty_chunks(self):
        with patch("answer_generation.get_openai_client") as mock_client:
            answer, sources = answer_with_chunks(
                "How are payments processed?",
                [],
            )

        self.assertEqual(
            answer,
            "I don't have enough information in the provided document.",
        )
        self.assertEqual(sources, [])
        mock_client.assert_not_called()

    def test_rejects_missing_chat_deployment(self):
        with patch("answer_generation.chat_deployment", None):
            with self.assertRaisesRegex(
                ValueError,
                "Azure OpenAI chat deployment is not configured."
            ):
                answer_with_chunks(
                    "How are payments processed?",
                    self.chunks,
                )

    @patch("answer_generation.verify_answer_claims")
    @patch("answer_generation.validate_citation_pages")
    @patch("answer_generation.get_openai_client")
    def test_generates_grounded_answer_successfully(
        self,
        mock_get_client,
        mock_validate,
        mock_verify,
    ):
        expected_answer = "Payments are processed through the gateway [page 2]."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=expected_answer),
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=None,
        )

        mock_get_client.return_value = mock_client

        mock_validate.return_value = {"valid": True}
        mock_verify.return_value = {
            "summary": "All claims supported",
            "results": [],
        }

        with patch("answer_generation.chat_deployment", "test-deployment"):
            answer, sources = answer_with_chunks(
                "How are payments processed?",
                self.chunks,
            )

        self.assertEqual(answer, expected_answer)
        self.assertEqual(sources, self.chunks)

        mock_client.chat.completions.create.assert_called_once()

        call_args = mock_client.chat.completions.create.call_args.kwargs

        self.assertEqual(call_args["model"], "test-deployment")
        self.assertEqual(len(call_args["messages"]), 2)
        self.assertIn("[Source: page 2]", call_args["messages"][1]["content"])
        self.assertIn("How are payments processed?", call_args["messages"][1]["content"])

        mock_validate.assert_called_once_with(
            expected_answer,
            self.chunks,
        )

        mock_verify.assert_called_once_with(
            expected_answer,
            self.chunks,
            client=mock_client,
            deployment="test-deployment",
        )

    @patch("answer_generation.verify_answer_claims")
    @patch("answer_generation.validate_citation_pages")
    @patch("answer_generation.get_openai_client")
    def test_applies_answer_transform_before_validation(
        self,
        mock_get_client,
        mock_validate,
        mock_verify,
    ):
        original_answer = "Payment processing information."
        transformed_answer = "Corrected payment processing information."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=original_answer),
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=None,
        )

        mock_get_client.return_value = mock_client
        mock_validate.return_value = {"valid": True}
        mock_verify.return_value = {
            "summary": "All claims supported",
            "results": [],
        }

        with patch("answer_generation.chat_deployment", "test-deployment"):
            answer, sources = answer_with_chunks(
                "How are payments processed?",
                self.chunks,
                answer_transform=lambda text: transformed_answer,
            )

        self.assertEqual(answer, transformed_answer)
        self.assertEqual(sources, self.chunks)

        mock_validate.assert_called_once_with(
            transformed_answer,
            self.chunks,
        )

        mock_verify.assert_called_once_with(
            transformed_answer,
            self.chunks,
            client=mock_client,
            deployment="test-deployment",
        )

    @patch("answer_generation.get_openai_client")
    def test_propagates_openai_api_failure(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "Azure OpenAI API unavailable"
        )

        mock_get_client.return_value = mock_client

        with patch("answer_generation.chat_deployment", "test-deployment"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Azure OpenAI API unavailable"
            ):
                answer_with_chunks(
                    "How are payments processed?",
                    self.chunks,
                )

    def test_openai_client_is_cached(self):
        import os

        from answer_generation import get_openai_client

        get_openai_client.cache_clear()

        try:
            with patch.dict(os.environ, {
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_API_VERSION": "2025-01-01",
            }):
                with patch("answer_generation.AzureOpenAI") as mock_azure:
                    first_client = get_openai_client()
                    second_client = get_openai_client()

                    self.assertIs(first_client, second_client)
                    mock_azure.assert_called_once()
        finally:
            get_openai_client.cache_clear()

    def test_diagnostic_logging_when_enabled(self):
        from answer_generation import diagnostic_log

        with patch("answer_generation.DIAGNOSTICS_ENABLED", True):
            with patch("answer_generation.logger.warning") as mock_warning:
                diagnostic_log(
                    "test_event",
                    document_hash="abc123",
                )

        mock_warning.assert_called_once()

    @patch("answer_generation.verify_answer_claims")
    @patch("answer_generation.validate_citation_pages")
    @patch("answer_generation.get_openai_client")
    def test_logs_citation_and_unsupported_claim_warnings(
        self,
        mock_get_client,
        mock_validate,
        mock_verify,
    ):
        mock_client = MagicMock()

        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Payment processing is completed [page 2]."
                    ),
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=None,
        )

        mock_get_client.return_value = mock_client

        mock_validate.return_value = {
            "valid": False,
            "errors": ["Invalid citation"],
        }

        mock_verify.return_value = {
            "summary": "One unsupported claim",
            "results": [
                {
                    "claim": "Payment processing is completed.",
                    "cited_pages": [2],
                    "verdict": "unsupported",
                }
            ],
        }

        with patch("answer_generation.chat_deployment", "test-deployment"):
            with patch("builtins.print") as mock_print:
                answer, sources = answer_with_chunks(
                    "How are payments processed?",
                    self.chunks,
                )

        self.assertTrue(answer)
        self.assertEqual(sources, self.chunks)

        printed_messages = [
            str(call.args[0])
            for call in mock_print.call_args_list
        ]

        self.assertIn(
            "Citation validation warning:",
            printed_messages,
        )
        self.assertIn(
            "Claim verification warning:",
            printed_messages,
        )

if __name__ == "__main__":
    unittest.main()
