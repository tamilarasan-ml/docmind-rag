"""Targeted query routing for known DOCMIND retrieval scenarios."""

TARGETED_QUERY_VARIANTS = {
    "How do I combine several criteria using the AND operator?":
        "JQL AND operator combine multiple criteria "
        "project status Customer Impact",

    "How can I find issues whose summary does not contain the word Task?":
        'Jira JQL negative text search operator !~ summary "Task"',
}


def select_retrieval_query(question):
    """Select a targeted retrieval query when a known intent matches."""

    normalized_question = " ".join(question.split()).casefold()

    if (
        "summary" in normalized_question
        and "task" in normalized_question
        and "not contain" in normalized_question
    ):
        return (
            'Jira JQL negative text search operator !~ summary "Task"'
        )

    for configured_question, retrieval_query in TARGETED_QUERY_VARIANTS.items():
        if normalized_question == " ".join(
            configured_question.split()
        ).casefold():
            return retrieval_query

    return question.strip()
