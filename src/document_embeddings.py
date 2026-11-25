
import math

MAX_EMBEDDING_CHUNKS = 100
EMBEDDING_BATCH_SIZE = 20
EMBEDDING_DIMENSIONS = 1536


def _get_response_value(item, key):
    """Support both SDK response objects and dictionary responses."""
    if isinstance(item, dict):
        return item.get(key)

    return getattr(item, key, None)


def _validate_embedding(embedding):
    """Validate an individual embedding vector."""

    if embedding is None:
        raise ValueError(
            "Embedding generation returned an empty vector."
        )

    try:
        if len(embedding) == 0:
            raise ValueError(
                "Embedding generation returned an empty vector."
            )

        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                "Unexpected embedding dimensions. "
                f"Expected {EMBEDDING_DIMENSIONS}."
            )

        valid_values = all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in embedding
        )

    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            ("Embedding generation", "Unexpected embedding")
        ):
            raise

        raise ValueError(
            "Embedding contains invalid numerical values."
        ) from exc

    if not valid_values:
        raise ValueError(
            "Embedding contains invalid numerical values."
        )

    if not any(value != 0 for value in embedding):
        raise ValueError(
            "Embedding vector cannot contain only zeros."
        )

    return embedding


def _get_ordered_batch_embeddings(response, batch_size):
    """Validate and order one batch using response indexes."""

    if hasattr(response, "data"):
        response = response.data

    if not isinstance(response, (list, tuple)):
        raise ValueError(
            "Embedding API returned an invalid batch response."
        )

    if len(response) != batch_size:
        raise ValueError(
            "Embedding API returned an unexpected number of vectors."
        )

    ordered = [None] * batch_size

    for item in response:
        index = _get_response_value(item, "index")
        embedding = _get_response_value(item, "embedding")

        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= batch_size
        ):
            raise ValueError(
                "Embedding API returned an invalid response index."
            )

        if ordered[index] is not None:
            raise ValueError(
                "Embedding API returned duplicate response indexes."
            )

        ordered[index] = _validate_embedding(embedding)

    if any(embedding is None for embedding in ordered):
        raise ValueError(
            "Embedding API returned missing response indexes."
        )

    return ordered


def embed_document_chunks(chunks, embedding_fn=None):
    """
    Generate embeddings for document chunks in memory.

    Default Azure path uses batches of 20.
    Custom embedding_fn retains the original sequential interface.
    """

    if not chunks:
        raise ValueError("No document chunks were provided.")

    if len(chunks) > MAX_EMBEDDING_CHUNKS:
        raise ValueError(
            f"The document exceeds the "
            f"{MAX_EMBEDDING_CHUNKS}-chunk embedding limit."
        )

    texts = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if not text:
            raise ValueError(
                "A document chunk contains empty text."
            )

        texts.append(text)

    embedded_chunks = []

    # Preserve existing custom embedding function behaviour.
    if embedding_fn is not None:
        for chunk, text in zip(chunks, texts):
            embedding = _validate_embedding(
                embedding_fn(text)
            )

            embedded_chunks.append({
                **chunk,
                "embedding": embedding,
            })

        return embedded_chunks

    # Default Azure OpenAI batch embedding path.
    from embed import get_embeddings

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch_texts = texts[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        response = get_embeddings(batch_texts)

        ordered_embeddings = _get_ordered_batch_embeddings(
            response,
            len(batch_texts),
        )

        for offset, embedding in enumerate(ordered_embeddings):
            chunk = chunks[start + offset]

            embedded_chunks.append({
                **chunk,
                "embedding": embedding,
            })

    return embedded_chunks
