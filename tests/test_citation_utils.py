
import unittest

from src.citation_utils import (
    extract_cited_pages,
    check_citation_pages_retrieved,
)


class TestCitationUtils(unittest.TestCase):

    def test_single_citation(self):
        self.assertEqual(
            extract_cited_pages("See [page 5]."),
            [5],
        )

    def test_multiple_pages_in_one_bracket(self):
        self.assertEqual(
            extract_cited_pages(
                "Insufficient information [page 2, page 12, page 13]."
            ),
            [2, 12, 13],
        )

    def test_separate_citations(self):
        self.assertEqual(
            extract_cited_pages(
                "Regex support [page 7][page 6]."
            ),
            [6, 7],
        )

    def test_duplicate_pages_are_removed(self):
        self.assertEqual(
            extract_cited_pages(
                "See [page 6] and [page 6]."
            ),
            [6],
        )

    def test_all_cited_pages_were_retrieved(self):
        result = check_citation_pages_retrieved(
            "See [page 6][page 7].",
            [{"page": 6}, {"page": 7}],
        )

        self.assertEqual(result["cited_pages"], [6, 7])
        self.assertTrue(result["citation_pages_retrieved"])

    def test_missing_retrieved_page(self):
        result = check_citation_pages_retrieved(
            "See [page 6][page 14].",
            [{"page": 6}, {"page": 7}],
        )

        self.assertFalse(result["citation_pages_retrieved"])

    def test_no_citations(self):
        result = check_citation_pages_retrieved(
            "The answer contains no citations.",
            [{"page": 6}],
        )

        self.assertEqual(result["cited_pages"], [])
        self.assertFalse(result["citation_pages_retrieved"])


if __name__ == "__main__":
    unittest.main()