
import numpy as np


def retrieve_document_chunks(
    question,
    embedded_chunks,
    top_k=3,
    embedding_fn=None,
):
    """
    Retrieve the most relevant chunks from one uploaded document.

    Uses cosine similarity against in-memory chunk embeddings.
    Does not access Azure AI Search or persistent storage.
    """

    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    if not embedded_chunks:
        raise ValueError("No embedded document chunks are available.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if embedding_fn is None:
        from embed import get_embedding
        embedding_fn = get_embedding

    query_embedding = np.asarray(
        embedding_fn(question.strip()),
        dtype=np.float32,
    )

    if query_embedding.ndim != 1 or query_embedding.size == 0:
        raise ValueError("Invalid query embedding.")

    query_norm = np.linalg.norm(query_embedding)

    if not np.isfinite(query_embedding).all() or query_norm == 0:
        raise ValueError("Query embedding is invalid.")

    scored_chunks = []

    for chunk in embedded_chunks:
        if "embedding" not in chunk:
            raise ValueError("A document chunk has no embedding.")

        chunk_embedding = np.asarray(
            chunk["embedding"],
            dtype=np.float32,
        )

        if (
            chunk_embedding.ndim != 1
            or chunk_embedding.shape != query_embedding.shape
        ):
            raise ValueError("Document embedding dimensions do not match.")

        chunk_norm = np.linalg.norm(chunk_embedding)

        if (
            not np.isfinite(chunk_embedding).all()
            or chunk_norm == 0
        ):
            raise ValueError("A document embedding is invalid.")

        similarity = float(
            np.dot(query_embedding, chunk_embedding)
            / (query_norm * chunk_norm)
        )

        scored_chunks.append({
            **chunk,
            "similarity": similarity,
        })

    scored_chunks.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return scored_chunks[:top_k]
