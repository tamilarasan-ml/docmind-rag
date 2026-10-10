
import re


def extract_cited_pages(answer):
    """
    Extract page numbers from citations such as:
      page 3
      pages 2-4
      [pages 8, 9, 15]
      [pages 2-4]
    """

    pages = set()

    pattern = (
        r"\bpages?\s+"
        r"(\d+(?:\s*[-–—]\s*\d+)?"
        r"(?:\s*,\s*\d+(?:\s*[-–—]\s*\d+)?)*)"
    )

    for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
        citation_text = match.group(1)

        for part in citation_text.split(","):
            part = part.strip()

            range_match = re.fullmatch(
                r"(\d+)\s*[-–—]\s*(\d+)",
                part,
            )

            if range_match:
                start, end = map(int, range_match.groups())

                if start <= end:
                    pages.update(range(start, end + 1))
            elif part.isdigit():
                pages.add(int(part))

    return sorted(pages)



def check_citation_pages_retrieved(answer, context_chunks):
    cited_pages = extract_cited_pages(answer)

    cited_pages = {int(page) for page in cited_pages}

    retrieved_pages = {
        int(chunk["page"])
        for chunk in context_chunks
        if chunk.get("page") is not None
    }

    return {
        "cited_pages": sorted(cited_pages),
        "citation_pages_retrieved": (
            bool(cited_pages)
            and cited_pages.issubset(retrieved_pages)
        ),
    }