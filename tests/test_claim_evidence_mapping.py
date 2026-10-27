
import unittest

from claim_verification import map_claims_to_evidence


class ClaimEvidenceMappingTests(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {"page": 5, "text": "AND combines criteria."},
            {"page": 11, "text": "now() returns the current timestamp."},
        ]

    def test_maps_claim_to_its_cited_page(self):
        result = map_claims_to_evidence(
            "AND combines criteria [page 5].",
            self.chunks,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["cited_pages"], [5])
        self.assertEqual(
            [item["page"] for item in result[0]["evidence"]],
            [5],
        )

    def test_does_not_attach_unrelated_page(self):
        result = map_claims_to_evidence(
            "AND combines criteria [page 5].",
            self.chunks,
        )

        evidence_pages = {
            item["page"] for item in result[0]["evidence"]
        }
        self.assertEqual(evidence_pages, {5})

    def test_uncited_claim_has_no_evidence(self):
        result = map_claims_to_evidence(
            "AND combines criteria.",
            self.chunks,
        )

        self.assertFalse(result[0]["has_citation"])
        self.assertEqual(result[0]["evidence"], [])

    def test_unretrieved_cited_page_has_no_evidence(self):
        result = map_claims_to_evidence(
            "AND combines criteria [page 99].",
            self.chunks,
        )

        self.assertEqual(result[0]["cited_pages"], [99])
        self.assertTrue(result[0]["has_citation"])
        self.assertFalse(result[0]["evidence_available"])


    def test_preserves_multiline_paragraph_with_trailing_citation(self):
        answer = (
            "The operator combines criteria.\n"
            "It requires all conditions to match [page 5]."
        )

        result = map_claims_to_evidence(answer, self.chunks)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["cited_pages"], [5])
        self.assertTrue(result[0]["evidence_available"])

    def test_maps_bullet_item_with_citation(self):
        answer = (
            "- AND combines criteria [page 5].\n"
            "- now() returns the current timestamp [page 11]."
        )

        result = map_claims_to_evidence(answer, self.chunks)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["cited_pages"], [5])
        self.assertEqual(result[1]["cited_pages"], [11])

    def test_uncited_bullet_remains_unverified(self):
        answer = (
            "- AND combines criteria [page 5].\n"
            "- This improves performance."
        )

        result = map_claims_to_evidence(answer, self.chunks)

        self.assertEqual(len(result), 2)
        self.assertTrue(result[0]["has_citation"])
        self.assertFalse(result[1]["has_citation"])


    def test_separates_sentences_with_different_citations(self):
        answer = (
            "AND combines criteria [page 5]. "
            "now() returns the current timestamp [page 11]."
        )

        result = map_claims_to_evidence(answer, self.chunks)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["cited_pages"], [5])
        self.assertEqual(result[1]["cited_pages"], [11])
        self.assertEqual(
            [item["page"] for item in result[0]["evidence"]],
            [5],
        )
        self.assertEqual(
            [item["page"] for item in result[1]["evidence"]],
            [11],
        )


    def test_keeps_uncited_operator_label_as_unverified(self):
        answer = (
            "- Equal to (=)\n"
            "  Matches issues where a field equals a value [page 5]."
        )

        result = map_claims_to_evidence(answer, self.chunks)

        # The label and explanation should be assessed as one unit.
        self.assertEqual(len(result), 1)
        self.assertIn("Equal to", result[0]["claim"])
        self.assertEqual(result[0]["cited_pages"], [5])
        self.assertTrue(result[0]["evidence_available"])

    def test_standalone_heading_is_not_a_factual_claim(self):
        answer = "Examples:"

        result = map_claims_to_evidence(answer, self.chunks)

        # A heading should not enter the claim-verification report.
        self.assertEqual(result, [])


    def test_recognizes_jql_section_heading(self):
        result = map_claims_to_evidence(
            "Text-search operators",
            self.chunks,
        )
        self.assertEqual(result, [])

    def test_recognizes_heading_ending_with_colon(self):
        result = map_claims_to_evidence(
            "Quick JQL syntax note:",
            self.chunks,
        )
        self.assertEqual(result, [])


    def test_recognizes_additional_jql_section_headings(self):
        headings = [
            "Basics",
            "Comparison / membership operators",
            "Null / special-state operators",
            "History / change operators",
            "Related operator pairs / alternatives",
        ]

        for heading in headings:
            with self.subTest(heading=heading):
                result = map_claims_to_evidence(
                    heading,
                    self.chunks,
                )
                self.assertEqual(result, [])






if __name__ == "__main__":
    unittest.main()
