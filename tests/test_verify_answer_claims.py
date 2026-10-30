
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


    def test_uncited_factual_claim_remains_uncertain(self):
        result = verify_answer_claims(
            "The system supports real-time payments.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 1)
        self.assertEqual(result["summary"]["supported"], 0)

    def test_uncited_conversational_text_is_identified(self):
        result = verify_answer_claims(
            "Let me know if you want more examples.",
            self.chunks,
        )

        # This test will initially fail until conversational
        # framing is explicitly handled.
        self.assertEqual(result["summary"]["uncertain"], 0)
        self.assertEqual(result["summary"]["supported"], 0)


    def test_uncited_introductory_framing_is_excluded(self):
        result = verify_answer_claims(
            "Here are the JQL operators shown in the provided "
            "documentation, with a short explanation.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)
        self.assertEqual(result["summary"]["supported"], 0)

    def test_uncited_evidence_backed_jql_summary_intro_is_excluded(self):
        result = verify_answer_claims(
            "Here's a concise, evidence-backed summary of the JQL "
            "operators shown in the provided context, what each does, "
            "and a source example for each.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)
        self.assertEqual(result["summary"]["supported"], 0)


    def test_uncited_closing_offer_is_excluded(self):
        result = verify_answer_claims(
            "If you want a specific example, tell me which fields "
            "and criteria you have and I'll help.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)
        self.assertEqual(result["summary"]["supported"], 0)

    def test_uncited_jql_summary_intro_is_excluded(self):
        result = verify_answer_claims(
            "Below is a concise summary of the JQL operators shown "
            "in the provided context, with each operator's purpose "
            "and the exact examples from the source.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)
        self.assertEqual(result["summary"]["supported"], 0)

    def test_uncited_operator_summary_intro_is_excluded(self):
        result = verify_answer_claims(
            "The provided context documents these JQL operators "
            "and how they are used. Below are the operators shown, "
            "with a brief purpose and a source example for each.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)

    def test_uncited_operator_closing_offer_is_excluded(self):
        result = verify_answer_claims(
            "If you want examples or explanations for a specific "
            "operator from the list above, tell me which one and "
            "I'll reproduce the documented example and description.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)

    def test_uncited_exact_example_guidance_is_excluded(self):
        result = verify_answer_claims(
            "If you want the exact example syntax preserved for any "
            "specific operator above, use the example shown with its "
            "cited page.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)
        self.assertEqual(result["summary"]["supported"], 0)

    def test_uncited_jql_catalog_intro_is_excluded(self):
        result = verify_answer_claims(
            "Here’s a concise catalog of JQL operators shown in the "
            "provided context, with what each operator means and a "
            "source example when available.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)
        self.assertEqual(result["summary"]["supported"], 0)


    def test_uncited_single_operator_closing_offer_is_excluded(self):
        result = verify_answer_claims(
            "If you want examples or more detail for any single "
            "operator above, tell me which one and I’ll show the "
            "exact example(s) from the context.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)
        self.assertEqual(result["summary"]["supported"], 0)

    def test_uncited_operator_clarification_offer_is_excluded(self):
        result = verify_answer_claims(
            "If you want examples or clarification for any specific "
            "operator above, tell me which one and I’ll show the exact "
            "example from the source.",
            self.chunks,
        )

        self.assertEqual(result["summary"]["uncertain"], 0)
        self.assertEqual(result["summary"]["supported"], 0)





    @patch("claim_verification.verify_claim_semantically")
    def test_exact_source_matched_jql_example_reaches_semantic_verifier(
        self, mock_verify
    ):
        source_chunks = [
            {
                "page": 4,
                "text": 'summary !~ "Task"',
            }
        ]

        answer = 'summary !~ "Task"'

        mock_verify.return_value = {
            "claim": answer,
            "verdict": "supported",
            "evidence": source_chunks,
            "explanation": "The exact JQL expression appears in the source.",
        }

        result = verify_answer_claims(
            answer,
            source_chunks,
        )

        # The semantic verifier must be called.
        mock_verify.assert_called_once()

        # The exact-matched source evidence must reach the verifier.
        passed_evidence = mock_verify.call_args.kwargs["evidence"]
        self.assertEqual(
            [item["page"] for item in passed_evidence],
            [4],
        )

        # The final verdict must come from semantic verification.
        self.assertEqual(result["summary"]["supported"], 1)
        self.assertEqual(result["summary"]["uncertain"], 0)

        verified_claim = result["results"][0]

        # Exact matching does not create an explicit citation.
        self.assertFalse(verified_claim["has_citation"])

        # Evidence association and semantic verification are separate.
        self.assertTrue(verified_claim["inherited_evidence"])
        self.assertEqual(verified_claim["verdict"], "supported")

if __name__ == "__main__":
    unittest.main()
