
import re
import sys
from pathlib import Path

# Allow imports from src when run from the repository root.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from chat import answer_question


QUESTION_ID = "q19"
QUESTION = "How can I use now() to find issues updated in the last two hours?"


def has_citation_for_page(text: str, page: int) -> bool:
    """Check for a citation such as [page 5] or [pages 5, 11]."""
    pattern = rf"\[pages?\s+[\d,\s]+\]"
    citations = re.findall(pattern, text, flags=re.IGNORECASE)

    for citation in citations:
        numbers = re.findall(r"\d+", citation)
        if str(page) in numbers:
            return True

    return False


def main() -> int:
    print(f"\nRunning targeted evaluation: {QUESTION_ID}")
    print(f"Question: {QUESTION}\n")

    answer, chunks = answer_question(QUESTION)
    answer = answer or ""

    print("Generated answer:")
    print(answer)

    retrieved_pages = sorted({
        chunk["page"]
        for chunk in chunks
        if "page" in chunk
    })
    print(f"\nRetrieved source pages: {retrieved_pages}")

    # Check the required answer content.
    checks = [
        (
            "Defines now() as the current timestamp",
            bool(
                 re.search(r"now\s*\(\)", answer, flags=re.IGNORECASE)
                 and re.search(r"\bcurrent timestamp\b", answer, flags=re.IGNORECASE)
            ),
        ),
        (
            "Includes the exact query updated > now(-2h)",
            "updated > now(-2h)" in answer,
        ),
        (
            "Explains the last-two-hours time window",
            bool(re.search(
                r"last two hours|two hours ago|two hours before now|updated since then",
                answer,
                flags=re.IGNORECASE,
            )),
        ),
        (
            "Cites page 11 for the now() definition",
            has_citation_for_page(answer, 11),
        ),
        (
            "Cites page 5 for the query example",
            has_citation_for_page(answer, 5),
        ),
    ]

    print("\nAutomated acceptance checks:")
    for description, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {description}")

    passed_count = sum(passed for _, passed in checks)
    total_count = len(checks)

    print(f"\nResult: {passed_count}/{total_count} checks passed.")

    if passed_count == total_count:
        print(f"{QUESTION_ID}: PASS")
        return 0

    print(f"{QUESTION_ID}: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
