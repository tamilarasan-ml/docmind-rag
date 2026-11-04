
import copy
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------
# Project paths and offline module isolation
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# evaluate.py imports answer_question from chat.py.
# Stub chat before importing evaluate to prevent Azure
# client initialisation through chat.py.

fake_chat = types.ModuleType("chat")
fake_chat.answer_question = MagicMock(
    name="mock_answer_question"
)

# Empty environment values prevent evaluate.py from
# constructing a live Azure OpenAI judge client.

offline_environment = {
    "AZURE_OPENAI_ENDPOINT": "",
    "AZURE_OPENAI_API_KEY": "",
    "AZURE_OPENAI_API_VERSION": "",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "",
}

with patch.dict(os.environ, offline_environment):
    with patch.dict(sys.modules, {"chat": fake_chat}):
        import evaluate as evaluation


# Import validate_report without executing the regression
# runner's main function.

REGRESSION_PATH = ROOT / "scripts" / "run_regression.py"

spec = importlib.util.spec_from_file_location(
    "docmind_run_regression",
    REGRESSION_PATH,
)

regression_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression_module)

validate_report = regression_module.validate_report


# ---------------------------------------------------------
# Context formatting tests
# ---------------------------------------------------------

class ContextFormattingTests(unittest.TestCase):

    def test_formats_page_and_text(self):
        chunks = [
            {"page": 5, "text": "First passage"},
            {"page": 8, "text": "Second passage"},
        ]

        result = evaluation.format_context(chunks)

        self.assertEqual(
            result,
            "[Source: page 5]\nFirst passage\n\n"
            "[Source: page 8]\nSecond passage",
        )

    def test_uses_content_when_text_is_missing(self):
        chunks = [
            {"page": 3, "content": "Fallback content"}
        ]

        result = evaluation.format_context(chunks)

        self.assertIn("Fallback content", result)

    def test_uses_unknown_page_when_missing(self):
        chunks = [{"text": "Example passage"}]

        result = evaluation.format_context(chunks)

        self.assertIn("[Source: page unknown]", result)

    def test_empty_context_returns_empty_string(self):
        self.assertEqual(
            evaluation.format_context([]),
            "",
        )


# ---------------------------------------------------------
# Retrieved page ranking tests
# ---------------------------------------------------------

class RetrievedPageRankingTests(unittest.TestCase):

    def test_preserves_retrieval_order(self):
        chunks = [
            {"page": 8},
            {"page": 3},
            {"page": 5},
        ]

        self.assertEqual(
            evaluation.get_retrieved_page_ranking(chunks),
            [8, 3, 5],
        )

    def test_preserves_duplicate_pages(self):
        chunks = [
            {"page": 5},
            {"page": 5},
            {"page": 8},
        ]

        self.assertEqual(
            evaluation.get_retrieved_page_ranking(chunks),
            [5, 5, 8],
        )

    def test_ignores_missing_and_invalid_pages(self):
        chunks = [
            {"page": 5},
            {},
            {"page": "invalid"},
            {"page": "8"},
        ]

        self.assertEqual(
            evaluation.get_retrieved_page_ranking(chunks),
            [5, 8],
        )


# ---------------------------------------------------------
# Judge verdict validation tests
# ---------------------------------------------------------

class VerdictValidationTests(unittest.TestCase):

    def setUp(self):
        self.valid_verdict = {
            "grounded": True,
            "relevant": True,
            "citation_supported": True,
            "appropriately_refused": False,
            "issues": "",
        }

    def test_accepts_valid_verdict(self):
        valid, error = evaluation.validate_verdict(
            self.valid_verdict
        )

        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_rejects_non_dictionary_verdict(self):
        valid, error = evaluation.validate_verdict("invalid")

        self.assertFalse(valid)
        self.assertEqual(
            error,
            "Judge output is not a JSON object.",
        )

    def test_rejects_missing_boolean_field(self):
        verdict = self.valid_verdict.copy()
        del verdict["grounded"]

        valid, error = evaluation.validate_verdict(verdict)

        self.assertFalse(valid)
        self.assertIn("grounded", error)

    def test_rejects_invalid_boolean_field(self):
        verdict = self.valid_verdict.copy()
        verdict["relevant"] = "yes"

        valid, error = evaluation.validate_verdict(verdict)

        self.assertFalse(valid)
        self.assertIn("relevant", error)

    def test_rejects_invalid_refusal_value(self):
        verdict = self.valid_verdict.copy()
        verdict["appropriately_refused"] = "yes"

        valid, error = evaluation.validate_verdict(verdict)

        self.assertFalse(valid)
        self.assertIn("appropriately_refused", error)

    def test_accepts_null_refusal_value(self):
        verdict = self.valid_verdict.copy()
        verdict["appropriately_refused"] = None

        valid, error = evaluation.validate_verdict(verdict)

        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_rejects_missing_issues_field(self):
        verdict = self.valid_verdict.copy()
        del verdict["issues"]

        valid, error = evaluation.validate_verdict(verdict)

        self.assertFalse(valid)
        self.assertIn("issues", error)

    def test_rejects_judge_error(self):
        verdict = {"error": "Simulated judge failure"}

        valid, error = evaluation.validate_verdict(verdict)

        self.assertFalse(valid)
        self.assertEqual(error, "Simulated judge failure")


# ---------------------------------------------------------
# Expected-fact coverage tests
# ---------------------------------------------------------

class FactCoverageTests(unittest.TestCase):

    def setUp(self):
        self.expected_facts = [
            "Fact A",
            "Fact B",
            "Fact C",
        ]

    def test_accepts_complete_fact_coverage(self):
        fact_results = [
            {
                "fact": "Fact A",
                "covered": True,
                "evidence": "Evidence A",
            },
            {
                "fact": "Fact B",
                "covered": True,
                "evidence": "Evidence B",
            },
            {
                "fact": "Fact C",
                "covered": True,
                "evidence": "Evidence C",
            },
        ]

        result = evaluation.calculate_fact_coverage(
            fact_results,
            self.expected_facts,
        )

        self.assertTrue(result["evaluated"])
        self.assertEqual(result["expected_count"], 3)
        self.assertEqual(result["covered_count"], 3)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["missing_facts"], [])
        self.assertIsNone(result["validation_error"])

    def test_calculates_partial_fact_coverage(self):
        fact_results = [
            {
                "fact": "Fact A",
                "covered": True,
                "evidence": "Evidence A",
            },
            {
                "fact": "Fact B",
                "covered": False,
                "evidence": "",
            },
            {
                "fact": "Fact C",
                "covered": True,
                "evidence": "Evidence C",
            },
        ]

        result = evaluation.calculate_fact_coverage(
            fact_results,
            self.expected_facts,
        )

        self.assertTrue(result["evaluated"])
        self.assertEqual(result["covered_count"], 2)
        self.assertEqual(result["coverage"], 0.6667)
        self.assertEqual(result["missing_facts"], ["Fact B"])

    def test_skips_evaluation_when_no_expected_facts(self):
        result = evaluation.calculate_fact_coverage([], [])

        self.assertFalse(result["evaluated"])
        self.assertEqual(result["expected_count"], 0)
        self.assertIsNone(result["coverage"])

    def test_rejects_non_list_fact_results(self):
        result = evaluation.calculate_fact_coverage(
            "invalid",
            self.expected_facts,
        )

        self.assertFalse(result["evaluated"])
        self.assertEqual(
            result["validation_error"],
            "fact_coverage is not a list.",
        )

    def test_rejects_fact_count_mismatch(self):
        result = evaluation.calculate_fact_coverage(
            [
                {
                    "fact": "Fact A",
                    "covered": True,
                    "evidence": "Evidence A",
                }
            ],
            self.expected_facts,
        )

        self.assertFalse(result["evaluated"])
        self.assertIn(
            "number of assessed facts",
            result["validation_error"],
        )

    def test_rejects_non_dictionary_fact_entry(self):
        result = evaluation.calculate_fact_coverage(
            ["invalid"],
            ["Fact A"],
        )

        self.assertFalse(result["evaluated"])
        self.assertIn(
            "not a JSON object",
            result["validation_error"],
        )

    def test_rejects_invalid_fact_label(self):
        result = evaluation.calculate_fact_coverage(
            [
                {
                    "fact": 123,
                    "covered": True,
                    "evidence": "Evidence",
                }
            ],
            ["Fact A"],
        )

        self.assertFalse(result["evaluated"])
        self.assertIn(
            "invalid fact",
            result["validation_error"],
        )

    def test_rejects_non_boolean_covered_value(self):
        result = evaluation.calculate_fact_coverage(
            [
                {
                    "fact": "Fact A",
                    "covered": "true",
                    "evidence": "Evidence",
                }
            ],
            ["Fact A"],
        )

        self.assertFalse(result["evaluated"])
        self.assertIn(
            "Invalid covered value",
            result["validation_error"],
        )

    def test_rejects_non_string_evidence(self):
        result = evaluation.calculate_fact_coverage(
            [
                {
                    "fact": "Fact A",
                    "covered": True,
                    "evidence": 123,
                }
            ],
            ["Fact A"],
        )

        self.assertFalse(result["evaluated"])
        self.assertIn(
            "Invalid evidence value",
            result["validation_error"],
        )

    def test_rejects_duplicate_fact_assessments(self):
        result = evaluation.calculate_fact_coverage(
            [
                {
                    "fact": "Fact A",
                    "covered": True,
                    "evidence": "Evidence A",
                },
                {
                    "fact": "Fact A",
                    "covered": True,
                    "evidence": "Duplicate evidence",
                },
            ],
            ["Fact A", "Fact B"],
        )

        self.assertFalse(result["evaluated"])
        self.assertIn(
            "Duplicate fact assessment",
            result["validation_error"],
        )

    def test_rejects_missing_and_unexpected_fact_labels(self):
        result = evaluation.calculate_fact_coverage(
            [
                {
                    "fact": "Fact A",
                    "covered": True,
                    "evidence": "Evidence A",
                },
                {
                    "fact": "Unexpected Fact",
                    "covered": True,
                    "evidence": "Unexpected evidence",
                },
            ],
            ["Fact A", "Fact B"],
        )

        self.assertFalse(result["evaluated"])
        self.assertEqual(
            result["missing_facts"],
            ["Fact B"],
        )
        self.assertIn(
            "Fact labels do not exactly match",
            result["validation_error"],
        )


# ---------------------------------------------------------
# Retrieval metric tests
# ---------------------------------------------------------

class RetrievalMetricsTests(unittest.TestCase):

    def test_calculates_hit_at_k_and_mrr(self):
        results = [
            {
                "expected_answerable": True,
                "gold_pages": [3],
                "retrieved_page_ranking": [8, 3, 5],
            },
            {
                "expected_answerable": True,
                "gold_pages": [3],
                "retrieved_page_ranking": [3, 4],
            },
        ]

        metrics = evaluation.calculate_retrieval_metrics(
            results
        )

        self.assertEqual(
            metrics["answerable_questions_evaluated"],
            2,
        )
        self.assertEqual(
            metrics["hit_at_k"]["hit_at_1"]["score"],
            0.5,
        )
        self.assertEqual(
            metrics["hit_at_k"]["hit_at_3"]["score"],
            1.0,
        )
        self.assertEqual(
            metrics["hit_at_k"]["hit_at_5"]["score"],
            1.0,
        )
        self.assertEqual(metrics["mrr"], 0.75)

    def test_returns_zero_hit_when_gold_page_not_retrieved(self):
        results = [
            {
                "expected_answerable": True,
                "gold_pages": [10],
                "retrieved_page_ranking": [1, 2, 3],
            }
        ]

        metrics = evaluation.calculate_retrieval_metrics(
            results
        )

        self.assertEqual(
            metrics["hit_at_k"]["hit_at_1"]["score"],
            0.0,
        )
        self.assertEqual(metrics["mrr"], 0.0)

    def test_excludes_unanswerable_trap_questions(self):
        results = [
            {
                "expected_answerable": False,
                "gold_pages": [3],
                "retrieved_page_ranking": [3],
            }
        ]

        metrics = evaluation.calculate_retrieval_metrics(
            results
        )

        self.assertEqual(
            metrics["answerable_questions_evaluated"],
            0,
        )
        self.assertIsNone(metrics["mrr"])

    def test_excludes_failed_evaluations(self):
        results = [
            {
                "expected_answerable": True,
                "gold_pages": [3],
                "retrieved_page_ranking": [3],
                "error": "Simulated evaluation failure",
            }
        ]

        metrics = evaluation.calculate_retrieval_metrics(
            results
        )

        self.assertEqual(
            metrics["answerable_questions_evaluated"],
            0,
        )

    def test_returns_empty_metrics_when_no_results(self):
        metrics = evaluation.calculate_retrieval_metrics([])

        self.assertEqual(
            metrics["answerable_questions_evaluated"],
            0,
        )
        self.assertEqual(metrics["hit_at_k"], {})
        self.assertIsNone(metrics["mrr"])


# ---------------------------------------------------------
# Regression report validation tests
# ---------------------------------------------------------

class RegressionReportValidationTests(unittest.TestCase):

    def setUp(self):
        self.gold_questions = [
            {
                "id": "q01",
                "type": "normal",
                "expected_answerable": True,
                "expected_facts": ["Fact A"],
            },
            {
                "id": "q02",
                "type": "trap",
                "expected_answerable": False,
                "expected_facts": [],
            },
        ]

        self.valid_report = {
            "summary": {
                "total_questions": 2,
                "valid_judge_results": 2,
                "evaluation_errors": 0,
                "questions_with_valid_fact_coverage": 1,
                "questions_not_evaluated_for_fact_coverage": 0,
                "expected_fact_count": 1,
                "covered_fact_count": 1,
                "unnecessary_refusals": 0,
                "answerable_not_refused_verdicts": 1,
                "correctly_refused_traps": 1,
                "failed_trap_refusals": 0,
                "indeterminate_trap_refusals": 0,
            },
            "results": [
                {
                    "id": "q01",
                    "expected_answerable": True,
                    "expected_facts": ["Fact A"],
                    "verdict": {
                        "grounded": True,
                        "relevant": True,
                        "citation_supported": True,
                        "citation_pages_retrieved": True,
                        "appropriately_refused": False,
                    },
                    "fact_coverage": {
                        "evaluated": True,
                    },
                },
                {
                    "id": "q02",
                    "expected_answerable": False,
                    "expected_facts": [],
                    "verdict": {
                        "grounded": True,
                        "relevant": True,
                        "citation_supported": True,
                        "citation_pages_retrieved": True,
                        "appropriately_refused": True,
                    },
                    "fact_coverage": {
                        "evaluated": False,
                    },
                },
            ],
        }

        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)

        self.report_path = (
            Path(self.temp_directory.name) / "candidate.json"
        )

    def validate(self, report):
        with self.report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(report, file)

        validate_report(
            self.report_path,
            self.gold_questions,
        )

    def test_accepts_valid_report(self):
        self.validate(copy.deepcopy(self.valid_report))

    def test_rejects_missing_question(self):
        report = copy.deepcopy(self.valid_report)
        report["results"].pop()

        with self.assertRaisesRegex(
            ValueError,
            "Question IDs",
        ):
            self.validate(report)

    def test_rejects_duplicate_question_ids(self):
        report = copy.deepcopy(self.valid_report)
        report["results"][1]["id"] = "q01"

        with self.assertRaisesRegex(
            ValueError,
            "Question IDs",
        ):
            self.validate(report)

    def test_rejects_incorrect_question_order(self):
        report = copy.deepcopy(self.valid_report)
        report["results"].reverse()

        with self.assertRaisesRegex(
            ValueError,
            "Question IDs",
        ):
            self.validate(report)

    def test_rejects_failed_groundedness(self):
        report = copy.deepcopy(self.valid_report)
        report["results"][0]["verdict"]["grounded"] = False

        with self.assertRaisesRegex(
            ValueError,
            "failed grounded",
        ):
            self.validate(report)

    def test_rejects_citation_page_retrieval_failure(self):
        report = copy.deepcopy(self.valid_report)
        report["results"][0]["verdict"][
            "citation_pages_retrieved"
        ] = False

        with self.assertRaisesRegex(
            ValueError,
            "citation page retrieval failure",
        ):
            self.validate(report)

    def test_rejects_incomplete_fact_coverage(self):
        report = copy.deepcopy(self.valid_report)
        report["summary"]["covered_fact_count"] = 0

        with self.assertRaisesRegex(
            ValueError,
            "expected facts were not covered",
        ):
            self.validate(report)

    def test_rejects_failed_trap_refusal(self):
        report = copy.deepcopy(self.valid_report)
        report["results"][1]["verdict"][
            "appropriately_refused"
        ] = False

        with self.assertRaisesRegex(
            ValueError,
            "correctly refused",
        ):
            self.validate(report)

    def test_rejects_unnecessary_answerable_refusal(self):
        report = copy.deepcopy(self.valid_report)
        report["results"][0]["verdict"][
            "appropriately_refused"
        ] = True

        with self.assertRaisesRegex(
            ValueError,
            "answerable but was marked",
        ):
            self.validate(report)


# ---------------------------------------------------------
# Test execution
# ---------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
