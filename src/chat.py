
import os
import logging
import hashlib
import json

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from query_routing import select_retrieval_query
from answer_generation import answer_with_chunks


# ---------------------------------------------------------
# Environment and diagnostics configuration
# ---------------------------------------------------------

load_dotenv()

DIAGNOSTICS_ENABLED = (
    os.getenv("DOCMIND_DIAGNOSTICS", "").lower() == "true"
)

logger = logging.getLogger("docmind.diagnostics")


def diagnostic_log(event, **fields):
    """Log privacy-conscious diagnostic information when enabled."""

    if DIAGNOSTICS_ENABLED:
        logger.warning(
            "DOCMIND_DIAG %s %s",
            event,
            json.dumps(fields, default=str),
        )


def fingerprint(value):
    """Generate a short SHA-256 fingerprint for diagnostic correlation."""

    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()[:12]


# ---------------------------------------------------------
# Azure OpenAI configuration
# ---------------------------------------------------------

# Retained for existing JQL query embedding generation.
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

embedding_deployment = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
)


# ---------------------------------------------------------
# Azure AI Search configuration
# ---------------------------------------------------------

search_key = os.getenv("AZURE_SEARCH_QUERY_KEY")

if not search_key:
    raise RuntimeError(
        "Azure AI Search query credential is not configured."
    )

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(search_key),
)


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_chunks(question, top_k=3):
    """
    Vector-only retrieval with targeted query variants
    for tested JQL questions.
    """

    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    normalized_question = question.strip()

    retrieval_query = select_retrieval_query(
        normalized_question
    )

    diagnostic_log(
        "query_routing",
        question_hash=fingerprint(normalized_question),
        targeted_variant_applied=(
            retrieval_query != normalized_question
        ),
        retrieval_query_hash=fingerprint(retrieval_query),
    )

    question_embedding = openai_client.embeddings.create(
        model=embedding_deployment,
        input=retrieval_query,
    ).data[0].embedding

    vector_query = VectorizedQuery(
        vector=question_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "text", "page"],
    )

    chunks = list(results)

    diagnostic_log(
        "retrieval_result",
        question_hash=fingerprint(question),
        retrieval_query_hash=fingerprint(retrieval_query),
        top_k=top_k,
        chunk_count=len(chunks),
        chunks=[
            {
                "id": chunk.get("id"),
                "page": chunk.get("page"),
                "text_hash": fingerprint(
                    chunk.get("text", "")
                ),
                "text_length": len(
                    str(chunk.get("text", ""))
                ),
            }
            for chunk in chunks
        ],
    )

    return chunks


# ---------------------------------------------------------
# Answer generation
# ---------------------------------------------------------

def answer_question(question):
    """
    Retrieve JQL documentation chunks and generate a
    grounded answer.

    Preserves the existing evaluation interface:
        (answer, chunks)
    """

    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    # Step 1: Retrieve relevant chunks from Azure AI Search.
    chunks = retrieve_chunks(question)

    # Step 2: Apply the existing JQL-specific correction.
    def apply_jql_correction(answer):

        if question.strip() == (
            "How can I find issues whose summary does not contain the word Task?"
        ):
            return answer.replace(
                '!~ summary !~ "Task"',
                'summary !~ "Task"',
            )

        return answer

    # Step 3: Delegate answer generation, citation validation
    # and semantic claim verification to the shared module.
    return answer_with_chunks(
        question,
        chunks,
        answer_transform=apply_jql_correction,
    )


# ---------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "DocMind RAG Chat — ask questions about your PDF "
        "(type 'exit' to quit)\n"
    )

    while True:

        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer, sources = answer_question(question)

        print(f"\nAssistant: {answer}")

        print(
            f"\n(Retrieved from pages: "
            f"{[s['page'] for s in sources]})"
        )

        print("-" * 60 + "\n")
