import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

load_dotenv()

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
)


def retrieve_chunks(question, top_k=3):
    """Same retrieval logic from Day 3."""
    question_embedding = openai_client.embeddings.create(
        model=embedding_deployment,
        input=question
    ).data[0].embedding

    vector_query = VectorizedQuery(
        vector=question_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "text", "page"]
    )

    return list(results)


def build_context(chunks):
    """
    Formats retrieved chunks into a labeled context block,
    so the model can cite exactly which chunk/page it used.
    """
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Source: page {chunk['page']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def answer_question(question):
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(question)
    context = build_context(chunks)

    # Step 2: Build a grounded prompt
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "If the answer isn't in the context, say you don't have enough information — "
        "do not use outside knowledge. Always cite the page number(s) you used, like [page 2]."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # Step 3: Generate the grounded answer
    response = openai_client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content, chunks


if __name__ == "__main__":
    print("DocMind RAG Chat — ask questions about your PDF (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break

        answer, sources = answer_question(question)

        print(f"\nAssistant: {answer}")
        print(f"\n(Retrieved from pages: {[s['page'] for s in sources]})")
        print("-" * 60 + "\n")