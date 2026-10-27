
import unittest

from citation_utils import extract_cited_pages
from citation_validation import validate_citation_pages


class CitationValidationTests(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {"page": 5, "text": "Document example"},
            {"page": 11, "text": "Function definition"},
        ]

    def test_extracts_single_and_multiple_page_citations(self):
        answer = (
            "The example is documented [page 5]. "
            "Definition [pages 5, 11]."
        )

        self.assertEqual(
            extract_cited_pages(answer),
            [5, 11],
        )

    def test_accepts_pages_present_in_retrieved_chunks(self):
        result = validate_citation_pages(
            "The example is documented [page 5].",
            self.chunks,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["cited_pages"], [5])
        self.assertEqual(result["invalid_pages"], [])

    def test_flags_a_page_not_retrieved(self):
        result = validate_citation_pages(
            "The answer cites an unavailable page [page 99].",
            self.chunks,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_pages"], [99])

    def test_flags_an_answer_without_citations(self):
        result = validate_citation_pages(
            "This answer contains no source citation.",
            self.chunks,
        )

        self.assertFalse(result["valid"])
        self.assertFalse(result["has_citations"])

    def test_accepts_page_ranges(self):
        result = validate_citation_pages(
            "The details are documented [pages 5-7].",
            [
                {"page": 5, "text": "Page five"},
                {"page": 6, "text": "Page six"},
                {"page": 7, "text": "Page seven"},
            ],
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["cited_pages"], [5, 6, 7])
        self.assertEqual(result["invalid_pages"], [])


if __name__ == "__main__":
    unittest.main()
