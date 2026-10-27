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

    expected_ids = [item["id"] for item in gold_questions]
    actual_ids = [item.get("id") for item in results]

    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("Question IDs, uniqueness, or ordering mismatch.")

    if summary.get("total_questions") != len(gold_questions):
        raise ValueError("Question count does not match gold dataset.")

    if summary.get("valid_judge_results") != len(gold_questions):
        raise ValueError("Not all questions have valid judge results.")

    if summary.get("evaluation_errors") != 0:
        raise ValueError("Evaluation contains errors.")

    for result in results:
        verdict = result.get("verdict", {})

        for field in ("grounded", "relevant", "citation_supported"):
            if verdict.get(field) is not True:
                raise ValueError(f"{result['id']} failed {field}.")

        if verdict.get("citation_pages_retrieved") is not True:
            raise ValueError(
                f"{result['id']} has a citation page retrieval failure."
            )

    configured_facts = sum(
        bool(item.get("expected_facts")) for item in gold_questions
    )

    if summary.get("questions_with_valid_fact_coverage") != configured_facts:
        raise ValueError("Expected-fact coverage is incomplete.")

    if summary.get("questions_not_evaluated_for_fact_coverage") != 0:
        raise ValueError("Some configured fact checks were not evaluated.")

    if summary.get("covered_fact_count") != summary.get("expected_fact_count"):
        raise ValueError("One or more expected facts were not covered.")

    for result in results:
        if result.get("expected_facts") and not result.get(
            "fact_coverage", {}
        ).get("evaluated"):
            raise ValueError(f"{result['id']} has invalid fact coverage.")

    gold_by_id = {item["id"]: item for item in gold_questions}
    answerable_refusals = []
    trap_refusals = []

    for result in results:
        expected_answerable = result.get("expected_answerable")
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

    # Run the targeted regression before spending time on the full suite.
    run_command([sys.executable, str(Q19_SCRIPT)])


    # Keep the candidate separate so failed runs are diagnosable.
    candidate_path = ROOT / "data" / ".eval_report_candidate.json"

    if candidate_path.exists():
        candidate_path.unlink()

    env = os.environ.copy()
    env["DOCMIND_EVAL_REPORT_PATH"] = str(candidate_path)

    # Run the full evaluation without overwriting the official report.
    run_command(
        [sys.executable, "-u", str(FULL_EVAL_SCRIPT)],
        env=env,
    )

    # Promote only after every validation check passes.
    validate_report(candidate_path, gold_questions)
    os.replace(candidate_path, REPORT_PATH)

    print(f"\nSUCCESS: validated report promoted to {REPORT_PATH}")



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nREGRESSION FAILED: {exc}", file=sys.stderr)
        print("The existing official report was not replaced.", file=sys.stderr)
        raise SystemExit(1)
