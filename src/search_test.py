import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

load_dotenv()

# Azure OpenAI client (to embed the question)
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Azure AI Search client
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
)


def search(question, top_k=3):
    # Step 1: turn the question into a vector, same way we embedded the chunks
    question_embedding = openai_client.embeddings.create(
        model=embedding_deployment,
        input=question
    ).data[0].embedding

    # Step 2: search the index for the most similar chunks
    vector_query = VectorizedQuery(
        vector=question_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )

    results = search_client.search(
        search_text=None,  # None = pure vector search (no keyword matching)
        vector_queries=[vector_query],
        select=["id", "text", "page"]
    )

    return list(results)


if __name__ == "__main__":
    question = "What is JQL and how do I use fields and operators?"
    print(f"Question: {question}\n")

    results = search(question)

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (page {r['page']}) ---")
        print(r["text"][:300])
        print()