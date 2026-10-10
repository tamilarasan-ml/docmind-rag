import json
import os
import re

from dotenv import load_dotenv
from openai import AzureOpenAI

from chat import answer_question
from citation_utils import check_citation_pages_retrieved


load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
JUDGE_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

GOLD_PATH = os.path.join("data", "eval_gold.json")
REPORT_PATH = os.path.join("data", "eval_report.json")


# ---------------------------------------------------------
# Azure OpenAI judge client
# ---------------------------------------------------------

client = None

if all(
    [
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_API_VERSION,
        JUDGE_DEPLOYMENT,
    ]
):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


# ---------------------------------------------------------
# Context formatting
# ---------------------------------------------------------

def format_context(context_chunks):
    """Format retrieved chunks for the LLM judge."""
    formatted_chunks = []

    for chunk in context_chunks:
        page = chunk.get("page", "unknown")
        content = chunk.get("text") or chunk.get("content") or ""
        formatted_chunks.append(f"[Source: page {page}]\n{content}")

    return "\n\n".join(formatted_chunks)


def get_retrieved_page_ranking(context_chunks):
    """
    Preserve the page order in the returned retrieval chunks.
    Repeated page numbers are intentionally retained because the
    retrieval results are chunk-ranked, not page-deduplicated.
    """
    ranking = []

    for chunk in context_chunks:
        page = chunk.get("page")
        if page is None:
            continue
        try:
            ranking.append(int(page))
        except (TypeError, ValueError):
            continue

    return ranking


# ---------------------------------------------------------
# Judge output validation
# ---------------------------------------------------------

def validate_verdict(verdict):
    """Validate the required fields in the judge response."""
    if not isinstance(verdict, dict):
        return False, "Judge output is not a JSON object."

    if verdict.get("error"):
        return False, str(verdict["error"])

    required_boolean_fields = [
        "grounded",
        "relevant",
        "citation_supported",
    ]

    for field in required_boolean_fields:
        if not isinstance(verdict.get(field), bool):
            return False, f"Missing or invalid boolean field: {field}"

    appropriately_refused = verdict.get("appropriately_refused")
    if appropriately_refused is not None and not isinstance(
        appropriately_refused, bool
    ):
        return False, "appropriately_refused must be true, false, or null."

    if not isinstance(verdict.get("issues"), str):
        return False, "Missing or invalid string field: issues"

    return True, None


# ---------------------------------------------------------
# Expected-fact coverage calculation
# ---------------------------------------------------------

def calculate_fact_coverage(fact_results, expected_facts):
    """
    Validate the judge's assessment of each expected fact.
    Coverage is a judge-based estimate, not proof of correctness.
    """
    if not expected_facts:
        return {
            "evaluated": False,
            "expected_count": 0,
            "covered_count": 0,
            "coverage": None,
            "missing_facts": [],
            "validation_error": None,
        }

    if not isinstance(fact_results, list):
        return {
            "evaluated": False,
            "expected_count": len(expected_facts),
            "covered_count": 0,
            "coverage": None,
            "missing_facts": [],
            "validation_error": "fact_coverage is not a list.",
        }

    if len(fact_results) != len(expected_facts):
        return {
            "evaluated": False,
            "expected_count": len(expected_facts),
            "covered_count": 0,
            "coverage": None,
            "missing_facts": [],
            "validation_error": (
                "The number of assessed facts does not match "
                "the number of expected facts."
            ),
        }

    normalized_results = {}

    for result in fact_results:
        if not isinstance(result, dict):
            return {
                "evaluated": False,
                "expected_count": len(expected_facts),
                "covered_count": 0,
                "coverage": None,
                "missing_facts": [],
                "validation_error": (
                    "A fact-coverage entry is not a JSON object."
                ),
            }

        fact = result.get("fact")
        covered = result.get("covered")
        evidence = result.get("evidence")

        if not isinstance(fact, str):
            error = "A fact-coverage entry has an invalid fact."
        elif not isinstance(covered, bool):
            error = f"Invalid covered value for fact: {fact}"
        elif not isinstance(evidence, str):
            error = f"Invalid evidence value for fact: {fact}"
        else:
            error = None

        if error:
            return {
                "evaluated": False,
                "expected_count": len(expected_facts),
                "covered_count": 0,
                "coverage": None,
                "missing_facts": [],
                "validation_error": error,
            }

        if fact in normalized_results:
            return {
                "evaluated": False,
                "expected_count": len(expected_facts),
                "covered_count": 0,
                "coverage": None,
                "missing_facts": [],
                "validation_error": f"Duplicate fact assessment: {fact}",
            }

        normalized_results[fact] = {
            "covered": covered,
            "evidence": evidence,
        }

    missing_assessments = [
        fact for fact in expected_facts if fact not in normalized_results
    ]
    unexpected_assessments = [
        fact for fact in normalized_results if fact not in expected_facts
    ]

    if missing_assessments or unexpected_assessments:
        return {
            "evaluated": False,
            "expected_count": len(expected_facts),
            "covered_count": 0,
            "coverage": None,
            "missing_facts": missing_assessments,
            "validation_error": (
                "Fact labels do not exactly match expected_facts."
            ),
        }

    covered_facts = [
        fact for fact in expected_facts
        if normalized_results[fact]["covered"]
    ]
    missing_facts = [
        fact for fact in expected_facts
        if not normalized_results[fact]["covered"]
    ]

    expected_count = len(expected_facts)
    covered_count = len(covered_facts)

    return {
        "evaluated": True,
        "expected_count": expected_count,
        "covered_count": covered_count,
        "coverage": round(covered_count / expected_count, 4),
        "missing_facts": missing_facts,
        "validation_error": None,
        "details": [
            {
                "fact": fact,
                "covered": normalized_results[fact]["covered"],
                "evidence": normalized_results[fact]["evidence"],
            }
            for fact in expected_facts
        ],
    }


# ---------------------------------------------------------
# LLM judge
# ---------------------------------------------------------

def judge_response(
    question,
    answer,
    context_chunks,
    expected_answerable=None,
    expected_facts=None,
):
    """Judge grounding, relevance, refusal, citations, and fact coverage."""
    if client is None:
        raise RuntimeError(
            "Azure OpenAI judge configuration is incomplete. "
            "Check AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_API_VERSION, and "
            "AZURE_OPENAI_CHAT_DEPLOYMENT in your .env file."
        )

    expected_facts = expected_facts or []
    context = format_context(context_chunks)

    if expected_answerable is True:
        expected_behavior = (
            "The question is answerable from the supplied source. "
            "The answer should address it using the source evidence."
        )
    elif expected_answerable is False:
        expected_behavior = (
            "The question is intentionally unanswerable from the "
            "supplied source. The answer should acknowledge the lack "
            "of supporting evidence rather than invent an answer."
        )
    else:
        expected_behavior = (
            "Determine whether the supplied context supports an answer. "
            "Do not assume the question is answerable."
        )

    if expected_facts and expected_answerable is not False:
        fact_guidance = f"""
Expected facts:
{json.dumps(expected_facts, ensure_ascii=False, indent=2)}

For each expected fact, independently determine whether the answer
covers that fact using the supplied context.

Rules:
- Assess the answer, not merely whether the fact appears in the context.
- A fact is covered only if the answer communicates its essential meaning.
- Do not mark a fact covered just because related words appear.
- Use the context to assess factual support.
- Do not reward unsupported claims.
- Return one assessment for every expected fact.
- Copy each fact string exactly as provided.
- Provide brief evidence explaining your decision.
"""
    else:
        fact_guidance = """
No expected facts have been supplied for this question.
Return an empty fact_coverage array.
Do not invent expected facts.
"""

    judge_prompt = f"""
You are an independent evaluator for a Retrieval-Augmented
Generation (RAG) question-answering system.

Evaluate the generated answer using only the supplied context
and the criteria below.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

RETRIEVED CONTEXT:
{context}

EXPECTED ANSWER BEHAVIOR:
{expected_behavior}

EVALUATION CRITERIA:

1. grounded:
   Are factual claims supported by the retrieved context?
   Unsupported or invented factual claims should not be considered grounded.

2. relevant:
   Does the answer directly address the question?

3. appropriately_refused:
   If the context does not support answering, did the answer acknowledge
   that limitation? If refusal is not applicable, return null.

4. citation_supported:
   Do the citations have semantic support in the retrieved context?
   A citation to a page that was not retrieved should not be treated as
   supported. The application separately checks page membership.

5. issues:
   Summarize specific problems. Use an empty string if there are no
   material issues.

6. fact_coverage:
   Follow the expected-fact instructions below.

{fact_guidance}

Return ONLY one valid JSON object. Do not use Markdown fences.

{{
  "grounded": true,
  "relevant": true,
  "appropriately_refused": null,
  "citation_supported": true,
  "issues": "",
  "fact_coverage": [
    {{
      "fact": "exact expected fact text",
      "covered": true,
      "evidence": "brief explanation based on the answer and context"
    }}
  ]
}}

For questions with no expected facts, return "fact_coverage": [].
All boolean fields must be actual JSON booleans, not strings.
Use null for appropriately_refused when it is not applicable.
"""

    response = client.chat.completions.create(
        model=JUDGE_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, evidence-based RAG evaluator. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": judge_prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content or ""
    cleaned_content = raw_content.strip()
    cleaned_content = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned_content,
        flags=re.IGNORECASE,
    )
    cleaned_content = re.sub(r"\s*```$", "", cleaned_content)

    try:
        verdict = json.loads(cleaned_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {exc}") from exc

    is_valid, validation_error = validate_verdict(verdict)
    if not is_valid:
        raise ValueError(f"Invalid judge verdict: {validation_error}")

    # Deterministic page-membership check; not semantic citation validation.
    verdict.update(
        check_citation_pages_retrieved(answer, context_chunks)
    )

    if not isinstance(verdict.get("fact_coverage"), list):
        verdict["fact_coverage"] = None

    return verdict


# ---------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------

def calculate_retrieval_metrics(results, k_values=(1, 3, 5)):
    """
    Calculate page-level Hit@K and MRR from original chunk ranking.

    A query is included only when it is answerable, has non-empty gold
    pages, has a preserved ranking, and did not fail evaluation.
    The rank is chunk rank; repeated pages remain repeated.
    """
    answerable = [
        result
        for result in results
        if result.get("expected_answerable") is True
        and isinstance(result.get("gold_pages"), list)
        and result["gold_pages"]
        and isinstance(result.get("retrieved_page_ranking"), list)
        and not result.get("error")
    ]

    if not answerable:
        return {
            "answerable_questions_evaluated": 0,
            "hit_at_k": {},
            "mrr": None,
        }

    hit_counts = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for result in answerable:
        gold_pages = {int(page) for page in result["gold_pages"]}
        ranking = [int(page) for page in result["retrieved_page_ranking"]]

        first_relevant_rank = next(
            (
                rank
                for rank, page in enumerate(ranking, start=1)
                if page in gold_pages
            ),
            None,
        )

        reciprocal_ranks.append(
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        for k in k_values:
            if any(page in gold_pages for page in ranking[:k]):
                hit_counts[k] += 1

    total = len(answerable)

    return {
        "answerable_questions_evaluated": total,
        "hit_at_k": {
            f"hit_at_{k}": {
                "hits": hit_counts[k],
                "total": total,
                "score": round(hit_counts[k] / total, 4),
            }
            for k in k_values
        },
        "mrr": round(sum(reciprocal_ranks) / total, 4),
    }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():
    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(
            f"Evaluation gold file not found: {GOLD_PATH}"
        )

    with open(GOLD_PATH, "r", encoding="utf-8") as file:
        gold_data = json.load(file)

    results = []

    for item in gold_data:
        question_id = item.get("id", "unknown")
        question_type = item.get("type", "unknown")
        question = item["question"]
        expected_answerable = item.get("expected_answerable")

        # Existing gold schema uses type="trap" and otherwise answerable.
        if expected_answerable is None:
            expected_answerable = question_type != "trap"

        expected_facts = item.get("expected_facts", [])
        if not isinstance(expected_facts, list):
            print(
                f"[!] {question_id}: expected_facts must be a list. "
                "Skipping fact coverage for this record."
            )
            expected_facts = []

        print(f"\nEvaluating {question_id}: {question}")

        try:
            # Keep production answer-generation path unchanged.
            answer, chunks = answer_question(question)

            # Preserve retrieval rank BEFORE sorting/deduplicating pages.
            retrieved_page_ranking = get_retrieved_page_ranking(chunks)

            verdict = judge_response(
                question=question,
                answer=answer,
                context_chunks=chunks,
                expected_answerable=expected_answerable,
                expected_facts=expected_facts,
            )

            fact_coverage = calculate_fact_coverage(
                verdict.get("fact_coverage"),
                expected_facts,
            )

            result = {
                "id": question_id,
                "type": question_type,
                "question": question,
                "expected_answerable": expected_answerable,
                "gold_pages": item.get("gold_pages", []),
                "expected_facts": expected_facts,
                "answer": answer,
                # Original ordered page list, preserving repeated pages.
                "retrieved_page_ranking": retrieved_page_ranking,
                # Keep a sorted unique list for convenient reporting.
                "retrieved_pages": sorted(set(retrieved_page_ranking)),
                "verdict": verdict,
                "fact_coverage": fact_coverage,
            }

            print(
                f"  Grounded: {verdict['grounded']} | "
                f"Relevant: {verdict['relevant']} | "
                f"Judge citation support: {verdict['citation_supported']}"
            )
            print(
                "  Cited pages retrieved (mechanical): "
                f"{verdict.get('citation_pages_retrieved')}"
            )

            if fact_coverage["evaluated"]:
                print(
                    f"  Fact coverage: {fact_coverage['covered_count']}/"
                    f"{fact_coverage['expected_count']} "
                    f"({fact_coverage['coverage']:.0%})"
                )
            elif expected_facts:
                print(
                    "  Fact coverage not evaluated: "
                    f"{fact_coverage['validation_error']}"
                )
            else:
                print("  Fact coverage: not configured")

        except Exception as exc:
            print(f"  [!] Evaluation failed: {exc}")

            result = {
                "id": question_id,
                "type": question_type,
                "question": question,
                "expected_answerable": expected_answerable,
                "gold_pages": item.get("gold_pages", []),
                "expected_facts": expected_facts,
                "answer": None,
                "retrieved_page_ranking": [],
                "retrieved_pages": [],
                "verdict": {"error": str(exc)},
                "fact_coverage": {
                    "evaluated": False,
                    "expected_count": len(expected_facts),
                    "covered_count": 0,
                    "coverage": None,
                    "missing_facts": [],
                    "validation_error": (
                        "Evaluation failed before fact coverage "
                        "could be calculated."
                    ),
                },
            }

        results.append(result)

    # -----------------------------------------------------
    # Summary metrics
    # -----------------------------------------------------

    total = len(results)

    successful_results = [
        result for result in results
        if "error" not in result.get("verdict", {})
    ]
    failed_results = [
        result for result in results
        if "error" in result.get("verdict", {})
    ]

    valid_judge_count = len(successful_results)

    grounded_count = sum(
        result["verdict"].get("grounded") is True
        for result in successful_results
    )
    relevant_count = sum(
        result["verdict"].get("relevant") is True
        for result in successful_results
    )
    judge_citation_supported_count = sum(
        result["verdict"].get("citation_supported") is True
        for result in successful_results
    )
    citation_pages_retrieved_count = sum(
        result["verdict"].get("citation_pages_retrieved") is True
        for result in successful_results
    )

    answerable_results = [
        result for result in successful_results
        if result["expected_answerable"] is True
    ]
    unanswerable_results = [
        result for result in successful_results
        if result["expected_answerable"] is False
    ]

    # These are judge verdict counts. On answerable questions, True means
    # the judge says a refusal was appropriate; False means it was not.
    unnecessary_refusals = sum(
        result["verdict"].get("appropriately_refused") is True
        for result in answerable_results
    )
    not_refused_as_expected = sum(
        result["verdict"].get("appropriately_refused") is False
        for result in answerable_results
    )
    correctly_refused_traps = sum(
        result["verdict"].get("appropriately_refused") is True
        for result in unanswerable_results
    )
    failed_trap_refusals = sum(
        result["verdict"].get("appropriately_refused") is False
        for result in unanswerable_results
    )
    indeterminate_trap_refusals = sum(
        result["verdict"].get("appropriately_refused") is None
        for result in unanswerable_results
    )

    fact_coverage_results = [
        result["fact_coverage"]
        for result in results
        if result["fact_coverage"].get("evaluated") is True
    ]
    configured_fact_questions = [
        result for result in results if result.get("expected_facts")
    ]
    fact_coverage_not_evaluated = (
        len(configured_fact_questions) - len(fact_coverage_results)
    )

    expected_fact_count = sum(
        item["expected_count"] for item in fact_coverage_results
    )
    covered_fact_count = sum(
        item["covered_count"] for item in fact_coverage_results
    )
    overall_fact_coverage = (
        round(covered_fact_count / expected_fact_count, 4)
        if expected_fact_count
        else None
    )

    # Retrieval metrics are independent of LLM-judge verdicts.
    retrieval_metrics = calculate_retrieval_metrics(results)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions: {total}")
    print(f"Valid judge results: {valid_judge_count}/{total}")
    print(f"Evaluation errors: {len(failed_results)}")

    if valid_judge_count:
        print(f"Grounded (LLM judge): {grounded_count}/{valid_judge_count}")
        print(f"Relevant (LLM judge): {relevant_count}/{valid_judge_count}")
        print(
            "Citation supported (LLM judge): "
            f"{judge_citation_supported_count}/{valid_judge_count}"
        )
        print(
            "Cited pages retrieved (mechanical): "
            f"{citation_pages_retrieved_count}/{valid_judge_count}"
        )

    print("\nRetrieval metrics (gold-page matching):")
    print(
        "Answerable questions evaluated for retrieval: "
        f"{retrieval_metrics['answerable_questions_evaluated']}"
    )
    for metric_name, metric in retrieval_metrics["hit_at_k"].items():
        print(
            f"{metric_name}: {metric['hits']}/{metric['total']} "
            f"({metric['score']:.1%})"
        )
    if retrieval_metrics["mrr"] is not None:
        print(f"MRR: {retrieval_metrics['mrr']:.4f}")
    else:
        print("MRR: not calculated")

    print("\nAnswerability and refusal:")
    print(f"Answerable questions evaluated: {len(answerable_results)}")
    print(
        "Unnecessary refusal verdicts: "
        f"{unnecessary_refusals}/{len(answerable_results)}"
    )
    print(
        "Answerable questions marked not-refused by judge: "
        f"{not_refused_as_expected}/{len(answerable_results)}"
    )
    print(f"Unanswerable questions evaluated: {len(unanswerable_results)}")
    print(
        f"Correctly refused traps: "
        f"{correctly_refused_traps}/{len(unanswerable_results)}"
    )
    print(
        f"Failed trap refusals: "
        f"{failed_trap_refusals}/{len(unanswerable_results)}"
    )
    print(
        f"Indeterminate trap refusals: "
        f"{indeterminate_trap_refusals}/{len(unanswerable_results)}"
    )

    print("\nExpected-fact coverage (LLM judge):")
    print(f"Questions with expected facts: {len(configured_fact_questions)}")
    print(f"Questions with valid fact coverage: {len(fact_coverage_results)}")
    print(
        "Questions not evaluated for fact coverage: "
        f"{fact_coverage_not_evaluated}"
    )
    if overall_fact_coverage is not None:
        print(
            f"Overall fact coverage: {covered_fact_count}/"
            f"{expected_fact_count} ({overall_fact_coverage:.1%})"
        )
    else:
        print("Overall fact coverage: not yet configured")

    # -----------------------------------------------------
    # Save full report
    # -----------------------------------------------------

    report = {
        "summary": {
            "total_questions": total,
            "valid_judge_results": valid_judge_count,
            "evaluation_errors": len(failed_results),
            "grounded": grounded_count,
            "relevant": relevant_count,
            # Explicit name distinguishes semantic judge from page membership.
            "judge_citation_supported": judge_citation_supported_count,
            "citation_pages_retrieved": citation_pages_retrieved_count,
            "retrieval_metrics": retrieval_metrics,
            "answerable_questions": len(answerable_results),
            "unnecessary_refusals": unnecessary_refusals,
            "answerable_not_refused_verdicts": not_refused_as_expected,
            "unanswerable_questions": len(unanswerable_results),
            "correctly_refused_traps": correctly_refused_traps,
            "failed_trap_refusals": failed_trap_refusals,
            "indeterminate_trap_refusals": indeterminate_trap_refusals,
            "questions_with_expected_facts": len(configured_fact_questions),
            "questions_with_valid_fact_coverage": len(fact_coverage_results),
            "questions_not_evaluated_for_fact_coverage": (
                fact_coverage_not_evaluated
            ),
            "expected_fact_count": expected_fact_count,
            "covered_fact_count": covered_fact_count,
            "overall_fact_coverage": overall_fact_coverage,
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
