
import re
from typing import Any


SAFE_REPLACEMENT = "[Potentially unsafe instruction omitted.]"


INJECTION_PATTERNS = {
    "instruction_override": [
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions\b",
        r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions\b",
        r"\bforget\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions\b",
        r"\boverride\s+(?:the\s+)?(?:system|developer)\s+instructions\b",
        r"\bfollow\s+these\s+instructions\s+instead\b",
        r"\byou\s+are\s+now\s+(?:a|an|the)\b",
    ],
    "confidentiality_attack": [
        r"\breveal\s+(?:my|your|the)\s+(?:system|developer)\s+(?:prompt|instructions?)\b",
        r"\bshow\s+(?:me\s+)?(?:your|the)\s+(?:system|developer)\s+(?:prompt|instructions?)\b",
        r"\bdisclose\s+(?:your|the)\s+(?:system|developer)\s+(?:prompt|instructions?)\b",
        r"\breveal\s+(?:the\s+)?(?:api\s+keys?|credentials|secrets)\b",
        r"\bprint\s+(?:the\s+)?(?:environment\s+variables|api\s+keys?|credentials)\b",
        r"\bdisclose\s+(?:the\s+)?(?:environment\s+variables|api\s+keys?|credentials)\b",
    ],
    "task_redirection": [
        r"\b(?:stop|ignore)\s+(?:answering|processing)\s+(?:the\s+)?(?:user'?s?\s+)?question\b",
        r"\bdo\s+not\s+answer\s+(?:the\s+)?(?:user'?s?\s+)?question\b",
        r"\binstead\s+of\s+(?:answering|following)\s+(?:the\s+)?(?:user'?s?\s+)?question\b",
        r"\bperform\s+the\s+following\s+unrelated\s+task\b",
    ],
}


COMPILED_PATTERNS = {
    category: [
        re.compile(pattern, re.IGNORECASE)
        for pattern in patterns
    ]
    for category, patterns in INJECTION_PATTERNS.items()
}


def _detect_categories(text: str) -> list[str]:
    """
    Return the categories of prompt-injection patterns detected.

    The original matched text is never returned.
    """

    if not text:
        return []

    detected = []

    for category, patterns in COMPILED_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            detected.append(category)

    return detected


def _sanitize_line(line: str) -> tuple[str, list[str]]:
    """
    Sanitize one line while preserving legitimate sentences.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        line,
    )

    sanitized_sentences = []
    detected_categories = set()

    for sentence in sentences:
        categories = _detect_categories(sentence)

        if categories:
            sanitized_sentences.append(SAFE_REPLACEMENT)
            detected_categories.update(categories)
        else:
            sanitized_sentences.append(sentence)

    return (
        " ".join(sanitized_sentences),
        sorted(detected_categories),
    )

def _sanitize_paragraph(paragraph: str) -> tuple[str, list[str]]:
    """
    Sanitize a paragraph while preserving bullet and numbered-list
    structure. Indented continuation lines belong to the preceding
    list item.
    """

    lines = paragraph.split("\n")
    output_lines = []
    detected_categories = set()

    ordinary_lines = []
    list_prefix = None
    list_content = []

    list_marker = re.compile(
        r"^(\s*(?:[-*•‣▪◦]|\d+[.)]|[A-Za-z][.)])\s+)(.*)$"
    )

    def flush_ordinary():
        if ordinary_lines:
            text = " ".join(
                line.strip() for line in ordinary_lines
            )

            sanitized_text, categories = _sanitize_line(text)

            output_lines.append(sanitized_text)
            detected_categories.update(categories)

            ordinary_lines.clear()

    def flush_list_item():
        nonlocal list_prefix, list_content

        if list_prefix is not None:
            content = " ".join(list_content)

            safe_content, categories = _sanitize_line(content)

            output_lines.append(list_prefix + safe_content)
            detected_categories.update(categories)

            list_prefix = None
            list_content = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        match = list_marker.match(line)

        if match:
            flush_ordinary()
            flush_list_item()

            list_prefix = match.group(1)
            list_content = [match.group(2).strip()]

        elif list_prefix is not None and line[:1].isspace():
            # Treat indented lines as continuation of the list item.
            list_content.append(stripped)

        else:
            flush_list_item()
            ordinary_lines.append(stripped)

    flush_ordinary()
    flush_list_item()

    return (
        "\n".join(output_lines),
        sorted(detected_categories),
    )



def prepare_safe_evidence(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Prepare sanitized evidence for model-facing prompts.

    Original chunks are not modified.

    Single line breaks are treated as PDF text wrapping.
    Blank lines preserve paragraph boundaries.

    Returns sanitized chunks and detection metadata without
    exposing the original detected instruction text.
    """

    if not chunks:
        return {
            "chunks": [],
            "detections": [],
        }

    safe_chunks = []
    detections = []

    for chunk in chunks:
        original_text = str(chunk.get("text", ""))
        page = chunk.get("page")

        # Normalize Windows and legacy line endings.
        normalized_text = (
            original_text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        # Preserve paragraph boundaries while joining PDF-wrapped lines.
        paragraphs = re.split(
            r"\n[ \t]*\n+",
            normalized_text,
        )

        sanitized_paragraphs = []
        detected_categories = set()

        for paragraph in paragraphs:
            sanitized_paragraph, categories = _sanitize_paragraph(
                paragraph
            )

            sanitized_paragraphs.append(sanitized_paragraph)
            detected_categories.update(categories)

        safe_text = "\n\n".join(sanitized_paragraphs)

        safe_chunks.append({
            "page": page,
            "text": safe_text,
        })

        if detected_categories:
            detections.append({
                "page": page,
                "categories": sorted(detected_categories),
            })

    return {
        "chunks": safe_chunks,
        "detections": detections,
    }

