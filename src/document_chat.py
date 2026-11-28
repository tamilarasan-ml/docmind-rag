
from document_retrieval import retrieve_document_chunks
from answer_generation import answer_with_chunks
from document_security import prepare_safe_evidence


def answer_uploaded_document(
    question,
    embedded_chunks,
    top_k=3,
):
    """
    Generate a grounded answer using only the supplied
    uploaded-document chunks.

    Retrieval is restricted to the supplied document.
    No Azure AI Search index is accessed.

    Original chunks are retained for citation traceability.
    Sanitized chunks are used for model-facing evidence.

    Returns:
        (answer, retrieved_chunks)
    """

    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    if not embedded_chunks:
        raise ValueError(
            "No uploaded document is available for answering."
        )

    # Step 1: Retrieve relevant chunks from this document only.
    retrieved_chunks = retrieve_document_chunks(
        question=question,
        embedded_chunks=embedded_chunks,
        top_k=top_k,
    )

    if not retrieved_chunks:
        return (
            "I don't have enough information in the provided document.",
            [],
        )

    # Step 2: Prepare sanitized evidence without modifying
    # the original retrieved chunks.
    security_result = prepare_safe_evidence(retrieved_chunks)
    safe_chunks = security_result["chunks"]

    # Step 3: Generate the answer using sanitized model context.
    # Original chunks remain available for citation validation
    # and source traceability.
    answer, sources = answer_with_chunks(
        question,
        retrieved_chunks,
        model_chunks=safe_chunks,
        safe_evidence=safe_chunks,
    )

    return answer, sources
