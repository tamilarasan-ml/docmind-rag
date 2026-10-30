
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

    def test_maps_now_function_explanation_to_separate_source_pages(self):
        chunks = [
            {
                "page": 5,
                "text": (
                    "updated > now(-2h) finds issues updated "
                    "in the last two hours."
                ),
            },
            {
                "page": 11,
                "text": "now() returns the current timestamp.",
            },
        ]

        answer = (
            "Use now() with a negative hour offset [page 5]. "
            "The now() function represents the current timestamp "
            "[page 11].\n\n"
            "For example:\n\n"
            "updated > now(-2h)\n\n"
            "This query finds issues updated in the last two hours "
            "[page 5]."
        )

        result = map_claims_to_evidence(answer, chunks)

        self.assertEqual(len(result), 4)

        self.assertEqual(result[0]["cited_pages"], [5])
        self.assertEqual(result[1]["cited_pages"], [11])

        self.assertTrue(result[2]["inherited_evidence"])
        self.assertEqual(result[2]["cited_pages"], [5])

        self.assertEqual(result[3]["cited_pages"], [5])

        for item, expected_page in zip(result, [5, 11, 5, 5]):
            self.assertTrue(item["evidence_available"])
            self.assertEqual(
                [evidence["page"] for evidence in item["evidence"]],
                [expected_page],
            )


    def test_keeps_uncited_operator_label_as_unverified(self):
        answer = (
            "- Equal to (=)\n"
            "  Matches issues where a field equals a value [page 5]."
        )

        result = map_claims_to_evidence(answer, self.chunks)

        self.assertEqual(len(result), 1)
        self.assertIn("Equal to", result[0]["claim"])
        self.assertEqual(result[0]["cited_pages"], [5])
        self.assertTrue(result[0]["evidence_available"])

    def test_standalone_heading_is_not_a_factual_claim(self):
        answer = "Examples:"

        result = map_claims_to_evidence(answer, self.chunks)

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
            "short definition",
            "common operators (with what they do and source examples)",
            "Basics",
            "Comparison / membership operators",
            "Null / special-state operators",
            "History / change operators",
            "Related operator pairs / alternatives",
            "Purpose",
            "How to use it",
            "Exact examples from the source",
            "Related operators",
            "Equality and inequality",
            "Set membership",
            "Comparisons (numbers/dates)",
            "IS / IS NOT (null / special states)",
            "Text-search (contains / does not contain)",
            "Change-history operators",
            "basic syntax",
            "comparison / equality operators",
            "range and boundary comparisons (numbers and dates)",
            "historical / change tracking",
            "notes and cross-references",
            "overview",
            "membership (collections)",
            "existence / special-state checks",
            "text searching (tilde)",
            "historical (was / was not)",
            "Brief definition",
            "How common JQL operators work (with source examples)",
            "How to build a simple query (recommended steps from the source)",
        ]

        for heading in headings:
            with self.subTest(heading=heading):
                result = map_claims_to_evidence(
                    heading,
                    self.chunks,
                )
                self.assertEqual(result, [])

    def test_maps_uncited_jql_example_when_exact_source_match_exists(self):
        chunks = [
            {
                "page": 9,
                "text": (
                    'project = "MYPROJECT" AND status = "Open" '
                    'AND "Customer Impact" = "High" '
                    "This query returns open issues."
                ),
            }
        ]

        answer = (
            'project = "MYPROJECT" AND status = "Open" '
            'AND "Customer Impact" = "High"'
        )

        result = map_claims_to_evidence(answer, chunks)

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["has_citation"])
        self.assertTrue(result[0]["evidence_available"])
        self.assertTrue(result[0]["inherited_evidence"])
        self.assertEqual(result[0]["cited_pages"], [9])

    def test_maps_nested_and_or_example_from_exact_source_match(self):
        chunks = [
            {
                "page": 9,
                "text": (
                    'project = "MYPROJECT" AND '
                    '(status = "Resolved" OR status = "Closed")'
                ),
            }
        ]

        answer = (
            'project = "MYPROJECT" AND '
            '(status = "Resolved" OR status = "Closed")'
        )

        result = map_claims_to_evidence(answer, chunks)

        self.assertTrue(result[0]["evidence_available"])
        self.assertTrue(result[0]["inherited_evidence"])
        self.assertEqual(result[0]["cited_pages"], [9])

    def test_maps_negative_text_search_example_from_exact_source_match(self):
        chunks = [
            {
                "page": 4,
                "text": 'summary !~ "Task"',
            }
        ]

        answer = 'summary !~ "Task"'

        result = map_claims_to_evidence(answer, chunks)

        self.assertTrue(result[0]["evidence_available"])
        self.assertTrue(result[0]["inherited_evidence"])
        self.assertEqual(result[0]["cited_pages"], [4])

    def test_does_not_inherit_evidence_for_unmatched_jql_example(self):
        chunks = [
            {
                "page": 9,
                "text": 'project = "MYPROJECT" AND status = "Open"',
            }
        ]

        answer = 'project = "OTHERPROJECT" AND status = "Closed"'

        result = map_claims_to_evidence(answer, chunks)

        self.assertFalse(result[0]["has_citation"])
        self.assertFalse(result[0]["evidence_available"])
        self.assertNotIn("inherited_evidence", result[0])


    def test_does_not_inherit_evidence_for_partial_value_match(self):
        chunks = [
            {
                "page": 12,
                "text": 'status = "Open Issues"',
            }
        ]

        answer = 'status = "Open"'

        result = map_claims_to_evidence(answer, chunks)

        self.assertFalse(result[0]["evidence_available"])
        self.assertNotIn("inherited_evidence", result[0])

    def test_does_not_inherit_evidence_from_larger_jql_expression(self):
        chunks = [
            {
                "page": 13,
                "text": (
                    'project = "MYPROJECT" AND '
                    'status = "Open"'
                ),
            }
        ]

        answer = 'status = "Open"'

        result = map_claims_to_evidence(answer, chunks)

        self.assertFalse(result[0]["evidence_available"])
        self.assertNotIn("inherited_evidence", result[0])

    def test_does_not_inherit_evidence_from_similar_field_name(self):
        chunks = [
            {
                "page": 14,
                "text": 'custom_summary = "Task"',
            }
        ]

        answer = 'summary = "Task"'

        result = map_claims_to_evidence(answer, chunks)

        self.assertFalse(result[0]["evidence_available"])
        self.assertNotIn("inherited_evidence", result[0])

    def test_maps_exact_jql_expression_with_different_whitespace(self):
        chunks = [
            {
                "page": 15,
                "text": (
                    'project   =   "MYPROJECT"   AND\n'
                    'status   =   "Open"'
                ),
            }
        ]

        answer = (
            'project = "MYPROJECT" AND '
            'status = "Open"'
        )

        result = map_claims_to_evidence(answer, chunks)

        self.assertTrue(result[0]["evidence_available"])
        self.assertTrue(result[0]["inherited_evidence"])
        self.assertEqual(result[0]["cited_pages"], [15])

    def test_does_not_inherit_evidence_when_match_precedes_and(self):
        chunks = [
            {
                "page": 16,
                "text": (
                    'status = "Open" AND '
                    'project = "MYPROJECT"'
                ),
            }
        ]

        answer = 'status = "Open"'

        result = map_claims_to_evidence(answer, chunks)

        self.assertFalse(result[0]["evidence_available"])
        self.assertNotIn("inherited_evidence", result[0])

    def test_does_not_inherit_evidence_when_match_precedes_or(self):
        chunks = [
            {
                "page": 17,
                "text": (
                    'status = "Open" OR '
                    'project = "MYPROJECT"'
                ),
            }
        ]

        answer = 'status = "Open"'

        result = map_claims_to_evidence(answer, chunks)

        self.assertFalse(result[0]["evidence_available"])
        self.assertNotIn("inherited_evidence", result[0])

    def test_preserves_case_of_quoted_jql_values(self):
        chunks = [
            {
                "page": 18,
                "text": 'summary = "Task"',
            }
        ]

        answer = 'summary = "task"'

        result = map_claims_to_evidence(answer, chunks)

        self.assertFalse(result[0]["evidence_available"])
        self.assertNotIn("inherited_evidence", result[0])

    def test_matches_jql_field_and_operator_case_insensitively(self):
        chunks = [
            {
                "page": 19,
                "text": 'SUMMARY = "Task"',
            }
        ]

        answer = 'summary = "Task"'

        result = map_claims_to_evidence(answer, chunks)

        self.assertTrue(result[0]["evidence_available"])
        self.assertTrue(result[0]["inherited_evidence"])
        self.assertEqual(result[0]["cited_pages"], [19])

    def test_matches_jql_keyword_operator_case_insensitively(self):
        chunks = [
        {
            "page": 20,
            "text": 'status IN ("Open", "Closed")',
        }
    ]

        answer = 'status in ("Open", "Closed")'

        result = map_claims_to_evidence(answer, chunks)

        self.assertTrue(result[0]["evidence_available"])
        self.assertTrue(result[0]["inherited_evidence"])
        self.assertEqual(result[0]["cited_pages"], [20])

if __name__ == "__main__":
    unittest.main()
