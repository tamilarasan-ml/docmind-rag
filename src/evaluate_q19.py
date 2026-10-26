
import json
import sys
from pathlib import Path

# Allow imports from the src directory when run from the repository root.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from chat import answer_question


QUESTION_ID = "q19"
QUESTION = "How can I use now() to find issues updated in the last two hours?"

EXPECTED_FACTS = [
    "The now() function represents the current timestamp.",
    "The source gives updated > now(-2h) as a query for issues updated in the last two hours.",
]


def main():
    print(f"\nRunning targeted evaluation: {QUESTION_ID}")
    print(f"Question: {QUESTION}\n")

    # Call the same answer-generation function used by the application.
    result = answer_question(QUESTION)

    # Display the result for manual verification.
    print("Generated answer:")
    print(result)

    print("\nExpected facts to verify:")
    for index, fact in enumerate(EXPECTED_FACTS, start=1):
        print(f"{index}. {fact}")

    print("\nManual acceptance checklist:")
    print("[ ] Answer explicitly defines now() as the current timestamp.")
    print("[ ] Answer includes updated > now(-2h).")
    print("[ ] Answer explains that the query finds issues updated in the last two hours.")
    print("[ ] Citations support the definition and the query.")
    

if __name__ == "__main__":
    main()
