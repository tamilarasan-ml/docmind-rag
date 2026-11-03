import unittest

from src.query_routing import select_retrieval_query


class QueryRoutingTests(unittest.TestCase):

    def test_routes_original_negative_summary_question(self):
        result = select_retrieval_query(
            "How can I find issues whose summary does not contain the word Task?"
        )

        self.assertIn("!~", result)
        self.assertIn("summary", result)

    def test_routes_uat_negative_summary_wording(self):
        result = select_retrieval_query(
            "How do I find Jira issues whose summary does not contain the word Task?"
        )

        self.assertIn("!~", result)
        self.assertIn("summary", result)

    def test_preserves_and_query_routing(self):
        result = select_retrieval_query(
            "How do I combine several criteria using the AND operator?"
        )

        self.assertIn("AND", result)
        self.assertIn("Customer Impact", result)

    def test_unrelated_question_remains_unchanged(self):
        question = "What is JQL?"

        self.assertEqual(
            select_retrieval_query(question),
            question,
        )


if __name__ == "__main__":
    unittest.main()
