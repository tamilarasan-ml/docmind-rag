
import unittest

from claim_verification import verify_claim_evidence


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


if __name__ == "__main__":
    unittest.main()


import unittest
from unittest.mock import MagicMock, patch

from claim_verification import verify_claim_semantically


class SemanticVerifierTests(unittest.TestCase):

    def setUp(self):
        self.evidence = [
            {
                "page": 5,
                "text": "Use the AND operator to combine criteria.",
            }
        ]

    @patch("claim_verification.AzureOpenAI")
    def test_returns_structured_supported_verdict(self, mock_client_class):
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
    def test_api_failure_returns_uncertain(self, mock_client_class):
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


if __name__ == "__main__":
    unittest.main()
