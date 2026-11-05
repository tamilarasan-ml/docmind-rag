import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "eval_gold.json"
REPORT_PATH = ROOT / "data" / "eval_report.json"
Q19_SCRIPT = ROOT / "src" / "evaluate_q19.py"
FULL_EVAL_SCRIPT = ROOT / "src" / "evaluate.py"
CANDIDATE_PATH = ROOT / "data" / ".eval_report_candidate.json"


def run_command(command, env=None):
    print(f"\n$ {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def validate_report(candidate_path, gold_questions):
    with candidate_path.open(encoding="utf-8") as file:
        report = json.load(file)

    summary = report.get("summary")
    results = report.get("results")

    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError("Report must contain summary and results.")

    # Validate gold dataset structure.
    if not isinstance(gold_questions, list) or not gold_questions:
        raise ValueError("Gold dataset must be a non-empty list.")

    expected_ids = []
    gold_by_id = {}

    for item in gold_questions:
        if not isinstance(item, dict):
            raise ValueError("Every gold question must be an object.")

        question_id = item.get("id")

        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("Gold question IDs must be non-empty strings.")

        if question_id in gold_by_id:
            raise ValueError(f"Duplicate gold question ID: {question_id}")

        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(
                f"{question_id}: gold question text must be non-empty."
            )

        question_type = item.get("type", "unknown")
        if not isinstance(question_type, str) or not question_type.strip():
            raise ValueError(f"{question_id}: invalid gold question type.")

        expected_facts = item.get("expected_facts", [])
        if (
            not isinstance(expected_facts, list)
            or any(
                not isinstance(fact, str) or not fact.strip()
                for fact in expected_facts
            )
            or len(expected_facts) != len(set(expected_facts))
        ):
            raise ValueError(
                f"{question_id}: gold expected_facts must contain "
                "unique, non-empty strings."
            )

        expected_answerable = item.get("expected_answerable")
        if (
            expected_answerable is not None
            and not isinstance(expected_answerable, bool)
        ):
            raise ValueError(
                f"{question_id}: gold expected_answerable must be Boolean."
            )

        gold_by_id[question_id] = item
        expected_ids.append(question_id)

    actual_ids = [item.get("id") if isinstance(item, dict) else None
                  for item in results]

    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("Question IDs, uniqueness, or ordering mismatch.")

    # Validate report metadata against the gold dataset.
    for result in results:
        question_id = result["id"]
        gold = gold_by_id[question_id]

        expected_type = gold.get("type", "unknown")
        if result.get("type") != expected_type:
            raise ValueError(
                f"{question_id}: question type does not match gold dataset."
            )

        gold_answerable = gold.get("expected_answerable")
        if gold_answerable is None:
            gold_answerable = expected_type != "trap"

        if result.get("expected_answerable") is not gold_answerable:
            raise ValueError(
                f"{question_id}: expected_answerable "
                "does not match gold dataset."
            )

        gold_facts = gold.get("expected_facts", [])
        if result.get("expected_facts") != gold_facts:
            raise ValueError(
                f"{question_id}: expected_facts do not match gold dataset."
            )

    if summary.get("total_questions") != len(gold_questions):
        raise ValueError("Question count does not match gold dataset.")

    if summary.get("valid_judge_results") != len(gold_questions):
        raise ValueError("Not all questions have valid judge results.")

    if summary.get("evaluation_errors") != 0:
        raise ValueError("Evaluation contains errors.")

    # Validate judge verdicts.
    for result in results:
        verdict = result.get("verdict")

        if not isinstance(verdict, dict):
            raise ValueError(f"{result['id']}: verdict must be an object.")

        for field in ("grounded", "relevant", "citation_supported"):
            if verdict.get(field) is not True:
                raise ValueError(f"{result['id']} failed {field}.")

        if verdict.get("citation_pages_retrieved") is not True:
            raise ValueError(
                f"{result['id']} has a citation page retrieval failure."
            )

    # Independently validate per-question fact coverage.
    expected_fact_count = 0
    covered_fact_count = 0
    questions_with_valid_fact_coverage = 0
    questions_not_evaluated_for_fact_coverage = 0

    for result in results:
        question_id = result["id"]
        expected_facts = result["expected_facts"]
        fact_coverage = result.get("fact_coverage")

        if not isinstance(fact_coverage, dict):
            raise ValueError(
                f"{question_id}: fact_coverage must be an object."
            )

        if not expected_facts:
            if (
                fact_coverage.get("evaluated") is not False
                or fact_coverage.get("expected_count") != 0
                or fact_coverage.get("covered_count") != 0
                or fact_coverage.get("coverage") is not None
                or fact_coverage.get("missing_facts") != []
                or fact_coverage.get("validation_error") is not None
                or fact_coverage.get("details", []) != []
            ):
                raise ValueError(
                    f"{question_id}: unexpected fact coverage "
                    "for a question without expected facts."
                )
            continue

        expected_count = len(expected_facts)
        expected_fact_count += expected_count

        if fact_coverage.get("evaluated") is not True:
            questions_not_evaluated_for_fact_coverage += 1
            raise ValueError(
                f"{question_id}: has invalid fact coverage."
            )

        if fact_coverage.get("validation_error") is not None:
            raise ValueError(
                f"{question_id}: fact coverage contains a validation error."
            )

        details = fact_coverage.get("details")

        if not isinstance(details, list):
            raise ValueError(
                f"{question_id}: fact coverage details must be a list."
            )

        detail_labels = []
        covered_facts = []
        missing_facts = []

        for detail in details:
            if not isinstance(detail, dict):
                raise ValueError(
                    f"{question_id}: invalid fact coverage detail."
                )

            fact = detail.get("fact")
            covered = detail.get("covered")
            evidence = detail.get("evidence")

            if not isinstance(fact, str):
                raise ValueError(
                    f"{question_id}: invalid fact label in coverage details."
                )

            if not isinstance(covered, bool):
                raise ValueError(
                    f"{question_id}: invalid covered value for {fact}."
                )

            if not isinstance(evidence, str):
                raise ValueError(
                    f"{question_id}: invalid evidence value for {fact}."
                )

            detail_labels.append(fact)

            if covered:
                covered_facts.append(fact)
            else:
                missing_facts.append(fact)

        if len(detail_labels) != len(set(detail_labels)):
            raise ValueError(
                f"{question_id}: duplicate fact coverage assessments."
            )

        if detail_labels != expected_facts:
            raise ValueError(
                f"{question_id}: fact coverage labels do not match "
                "expected_facts."
            )

        calculated_covered_count = len(covered_facts)
        calculated_coverage = round(
            calculated_covered_count / expected_count,
            4,
        )

        if fact_coverage.get("expected_count") != expected_count:
            raise ValueError(
                f"{question_id}: fact coverage expected_count mismatch."
            )

        if fact_coverage.get("covered_count") != calculated_covered_count:
            raise ValueError(
                f"{question_id}: fact coverage covered_count mismatch."
            )

        if fact_coverage.get("coverage") != calculated_coverage:
            raise ValueError(
                f"{question_id}: fact coverage percentage mismatch."
            )

        if fact_coverage.get("missing_facts") != missing_facts:
            raise ValueError(
                f"{question_id}: missing_facts does not match assessments."
            )

        questions_with_valid_fact_coverage += 1
        covered_fact_count += calculated_covered_count

    # Reconcile independently calculated totals with the report summary.
    if summary.get("questions_with_valid_fact_coverage") != (
        questions_with_valid_fact_coverage
    ):
        raise ValueError("Valid fact coverage question count mismatch.")

    if summary.get("questions_not_evaluated_for_fact_coverage") != (
        questions_not_evaluated_for_fact_coverage
    ):
        raise ValueError("Unevaluated fact coverage question count mismatch.")

    if summary.get("expected_fact_count") != expected_fact_count:
        raise ValueError(
            "Expected fact count does not match individual records."
        )

    if summary.get("covered_fact_count") != covered_fact_count:
        raise ValueError(
            "Covered fact count does not match individual records."
        )

    if covered_fact_count != expected_fact_count:
        raise ValueError("One or more expected facts were not covered.")

    # Validate refusal behavior against gold metadata.
    answerable_refusals = []
    trap_refusals = []

    for result in results:
        expected_answerable = result["expected_answerable"]
        expected_type = gold_by_id[result["id"]].get("type", "unknown")
        refusal = result.get("verdict", {}).get("appropriately_refused")

        if expected_type == "trap" or expected_answerable is False:
            trap_refusals.append(refusal)

            if refusal is not True:
                raise ValueError(
                    f"{result['id']} should have been correctly refused."
                )

        elif expected_answerable is True:
            answerable_refusals.append(refusal)

            if refusal is True:
                raise ValueError(
                    f"{result['id']} was answerable but was marked "
                    "as an appropriate refusal."
                )

    checks = {
        "unnecessary_refusals": sum(
            value is True for value in answerable_refusals
        ),
        "answerable_not_refused_verdicts": sum(
            value is False for value in answerable_refusals
        ),
        "correctly_refused_traps": sum(
            value is True for value in trap_refusals
        ),
        "failed_trap_refusals": sum(
            value is False for value in trap_refusals
        ),
        "indeterminate_trap_refusals": sum(
            value is None for value in trap_refusals
        ),
    }

    for key, expected in checks.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{key} mismatch: report={summary.get(key)}, "
                f"calculated={expected}."
            )

    print("\nCandidate report validation: PASS")
    print(f"Questions: {len(results)}/{len(gold_questions)}")
    print("Evaluation errors: 0")
    print("Grounded, relevant, citation support: all pass")
    print("Expected-fact coverage: complete")
    print("Refusal behavior: PASS")
    print(
        "Note: LLM-judge citation support is an automated signal, "
        "not proof of semantic correctness."
    )


def main():
    with GOLD_PATH.open(encoding="utf-8") as file:
        gold_questions = json.load(file)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove any stale candidate before starting any regression step.
    if CANDIDATE_PATH.exists():
        CANDIDATE_PATH.unlink()

    # Run the targeted regression before spending time on the full suite.
    run_command([sys.executable, str(Q19_SCRIPT)])

    # Keep the candidate separate so failed runs are diagnosable.
    env = os.environ.copy()
    env["DOCMIND_EVAL_REPORT_PATH"] = str(CANDIDATE_PATH)

    # Run the full evaluation without overwriting the official report.
    run_command(
        [sys.executable, "-u", str(FULL_EVAL_SCRIPT)],
        env=env,
    )

    # Promote only after every validation check passes.
    validate_report(CANDIDATE_PATH, gold_questions)
    os.replace(CANDIDATE_PATH, REPORT_PATH)

    print(f"\nSUCCESS: validated report promoted to {REPORT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nREGRESSION FAILED: {exc}", file=sys.stderr)
        print(
            "The existing official report was not replaced.",
            file=sys.stderr,
        )
        raise SystemExit(1)
