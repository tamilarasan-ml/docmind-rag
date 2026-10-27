
from typing import Any, Iterable

from citation_utils import extract_cited_pages


def validate_citation_pages(
    answer: str,
    chunks: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """
    Check whether every cited page appears in the retrieved chunks.

    This verifies page availability, not whether the page semantically
    supports the claim.
    """
    retrieved_pages = {
        int(chunk["page"])
        for chunk in chunks
        if chunk.get("page") is not None
    }

    cited_pages = extract_cited_pages(answer or "")
    invalid_pages = sorted(set(cited_pages) - retrieved_pages)

    return {
        "valid": bool(cited_pages) and not invalid_pages,
        "cited_pages": cited_pages,
        "retrieved_pages": sorted(retrieved_pages),
        "invalid_pages": invalid_pages,
        "has_citations": bool(cited_pages),
    }
