import math


def embed_document_chunks(chunks, embedding_fn=None):
    """
    Generate embeddings for document chunks in memory.

    Args:
        chunks: List of dictionaries containing id, text and page.
        embedding_fn: Optional embedding function for dependency
                      injection and unit testing.

    Returns:
        New list of chunks containing their embeddings.
    """

    if not chunks:
        raise ValueError("No document chunks were provided.")

    # Import lazily so tests can use a mocked embedding function
    # without initializing the Azure OpenAI client.
    if embedding_fn is None:
        from embed import get_embedding
        embedding_fn = get_embedding

    embedded_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if not text:
            raise ValueError(
                "A document chunk contains empty text."
            )

        embedding = embedding_fn(text)

        if embedding is None or len(embedding) == 0:
            raise ValueError(
                "Embedding generation returned an empty vector."
            )

        if len(embedding) != 1536:
            raise ValueError(
                "Unexpected embedding dimensions. "
                "Expected 1536."
            )

        try:
            valid_values = all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in embedding
            )
        except (TypeError, ValueError, OverflowError):
            valid_values = False

        if not valid_values:
            raise ValueError(
                "Embedding contains invalid numerical values."
            )

        if not any(value != 0 for value in embedding):
            raise ValueError(
                "Embedding vector cannot contain only zeros."
            )

        embedded_chunk = {
            **chunk,
            "embedding": embedding,
        }

        embedded_chunks.append(embedded_chunk)

    return embedded_chunks