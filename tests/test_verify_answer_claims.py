
import unittest
from unittest.mock import patch

from claim_verification import verify_answer_claims


class VerifyAnswerClaimsTests(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {"page": 5, "text": "AND combines criteria."},
        ]

    @patch("claim_verification.verify_claim_semantically")
    def test_verifies_claim_using_cited_page(self, mock_verify):
        mock_verify.return_value = {
            "claim": "AND combines criteria [page 5].",
            "verdict": "supported",
            "evidence": self.chunks,
            "explanation": "Directly supported.",
        }

        result = verify_answer_claims(
            "AND combines criteria [page 5].",
            self.chunks,
        )

        self.assertEqual(result["summary"]["supported"], 1)
        self.assertEqual(result["summary"]["unsupported"], 0)
        mock_verify.assert_called_once()

        passed_evidence = mock_verify.call_args.kwargs["evidence"]
        self.assertEqual(
            [item["page"] for item in passed_evidence],
            [5],
        )

    def test_uncited_claim_is_not_sent_to_verifier(self):
        result = verify_answer_claims(
            "AND combines criteria.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 1)
        self.assertEqual(result["summary"]["supported"], 0)


if __name__ == "__main__":
    unittest.main()
