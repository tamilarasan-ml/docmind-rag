
from document_retrieval import retrieve_document_chunks
from answer_generation import answer_with_chunks


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

    # Step 2: Generate a grounded answer using the shared engine.
    answer, sources = answer_with_chunks(
        question,
        retrieved_chunks,
    )

    return answer, sources
