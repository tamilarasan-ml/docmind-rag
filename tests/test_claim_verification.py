import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from claim_verification import (
    verify_answer_claims,
    verify_claim_evidence,
    verify_claim_semantically,
)


class ClaimVerificationTests(unittest.TestCase):

    def setUp(self):
        self.evidence = [
            {
                "page": 5,
                "text": "Use the AND operator to combine criteria.",
            }
        ]

    def test_supported_claim_is_recorded(self):
        result = verify_claim_evidence(
            "AND combines criteria.",
            self.evidence,
            "supported",
            "The evidence directly states this.",
        )

        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(result["evidence"][0]["page"], 5)

    def test_unsupported_claim_is_recorded(self):
        result = verify_claim_evidence(
            "AND always improves performance.",
            self.evidence,
            "unsupported",
        )

        self.assertEqual(result["verdict"], "unsupported")

    def test_uncertain_claim_is_recorded(self):
        result = verify_claim_evidence(
            "AND works in every possible query.",
            [],
            "uncertain",
        )

        self.assertEqual(result["verdict"], "uncertain")
        self.assertEqual(result["evidence"], [])

    def test_invalid_verdict_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_claim_evidence(
                "Example claim",
                self.evidence,
                "maybe",
            )


class SemanticVerifierTests(unittest.TestCase):

    def setUp(self):
        self.evidence = [
            {
                "page": 5,
                "text": "Use the AND operator to combine criteria.",
            }
        ]

    @patch("claim_verification.AzureOpenAI")
    def test_returns_structured_supported_verdict(
        self,
        mock_client_class,
    ):
        client = MagicMock()
        mock_client_class.return_value = client

        parsed = MagicMock()
        parsed.verdict = "supported"
        parsed.explanation = "The passage directly supports the claim."

        message = MagicMock()
        message.refusal = None
        message.parsed = parsed

        client.beta.chat.completions.parse.return_value.choices = [
            MagicMock(message=message)
        ]

        result = verify_claim_semantically(
            "AND combines criteria.",
            self.evidence,
            client=client,
            deployment="test-deployment",
        )

        self.assertEqual(result["verdict"], "supported")
        client.beta.chat.completions.parse.assert_called_once()

    def test_missing_evidence_returns_uncertain(self):
        result = verify_claim_semantically(
            "AND combines criteria.",
            [],
        )

        self.assertEqual(result["verdict"], "uncertain")

    @patch("claim_verification.AzureOpenAI")
    def test_api_failure_returns_uncertain(
        self,
        mock_client_class,
    ):
        client = MagicMock()
        client.beta.chat.completions.parse.side_effect = RuntimeError(
            "Simulated API failure"
        )

        result = verify_claim_semantically(
            "AND combines criteria.",
            self.evidence,
            client=client,
            deployment="test-deployment",
        )

        self.assertEqual(result["verdict"], "uncertain")

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-api-key",
            "AZURE_OPENAI_API_VERSION": "2024-10-21",
        },
    )
    @patch("claim_verification.AzureOpenAI")
    def test_initializes_default_azure_client(
        self,
        mock_client_class,
    ):
        client = MagicMock()
        mock_client_class.return_value = client

        parsed = MagicMock()
        parsed.verdict = "supported"
        parsed.explanation = "Evidence supports the claim."

        message = MagicMock()
        message.refusal = None
        message.parsed = parsed

        client.beta.chat.completions.parse.return_value.choices = [
            MagicMock(message=message)
        ]

        result = verify_claim_semantically(
            "AND combines criteria.",
            self.evidence,
            deployment="test-deployment",
        )

        self.assertEqual(result["verdict"], "supported")

        mock_client_class.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-api-key",
            api_version="2024-10-21",
        )

    @patch("claim_verification.AzureOpenAI")
    def test_returns_uncertain_when_verifier_refuses(
        self,
        mock_client_class,
    ):
        client = MagicMock()
        mock_client_class.return_value = client

        message = MagicMock()
        message.refusal = "I cannot assess this claim."
        message.parsed = None

        client.beta.chat.completions.parse.return_value.choices = [
            MagicMock(message=message)
        ]

        result = verify_claim_semantically(
            "AND combines criteria.",
            self.evidence,
            client=client,
            deployment="test-deployment",
        )

        self.assertEqual(result["verdict"], "uncertain")
        self.assertEqual(
            result["explanation"],
            "Verification could not be completed (ValueError).",
        )


class VerifyClaimsTests(unittest.TestCase):

    @patch("claim_verification.map_claims_to_evidence")
    def test_skips_standalone_heading(self, mock_mapping):
        mock_mapping.return_value = [
            {
                "claim": "Purpose",
                "cited_pages": [],
                "evidence": [],
                "has_citation": False,
                "evidence_available": False,
            }
        ]

        with patch(
            "claim_verification.verify_claim_semantically"
        ) as mock_verify:
            result = verify_answer_claims(
                "Purpose",
                [],
            )

        self.assertEqual(result["results"], [])
        self.assertEqual(
            result["summary"],
            {
                "supported": 0,
                "unsupported": 0,
                "uncertain": 0,
            },
        )

        mock_mapping.assert_called_once()
        mock_verify.assert_not_called()

    def test_returns_uncertain_when_cited_page_is_unavailable(self):
        answer = "AND combines criteria [page 11]."
        chunks = [
            {
                "page": 5,
                "text": "Use the AND operator to combine criteria.",
            }
        ]

        with patch(
            "claim_verification.verify_claim_semantically"
        ) as mock_verify:
            result = verify_answer_claims(answer, chunks)

        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(
            result["results"][0]["verdict"],
            "uncertain",
        )
        self.assertEqual(
            result["results"][0]["explanation"],
            "Cited page was not available in retrieved evidence.",
        )
        mock_verify.assert_not_called()

    @patch.dict("os.environ", {"DOCMIND_DIAGNOSTICS": "true"})
    @patch("claim_verification.verify_claim_semantically")
    def test_logs_diagnostics_without_exposing_claim_text(
        self,
        mock_verify,
    ):
        answer = "AND combines criteria [page 5]."
        chunks = [
            {
                "page": 5,
                "text": "Use the AND operator to combine criteria.",
            }
        ]

        mock_verify.return_value = {
            "verdict": "supported",
            "explanation": "Evidence supports the claim.",
        }

        output = io.StringIO()

        with redirect_stdout(output):
            result = verify_answer_claims(answer, chunks)

        logged_output = output.getvalue()

        self.assertEqual(
            result["results"][0]["verdict"],
            "supported",
        )
        self.assertIn(
            "DOCMIND_DIAG mapped_claims:",
            logged_output,
        )
        self.assertIn(
            "DOCMIND_DIAG claim_verdict:",
            logged_output,
        )
        self.assertIn("claim_hash", logged_output)
        self.assertIn("supported", logged_output)

        self.assertNotIn(answer, logged_output)
        self.assertNotIn(
            "Evidence supports the claim.",
            logged_output,
        )


if __name__ == "__main__":
    unittest.main()