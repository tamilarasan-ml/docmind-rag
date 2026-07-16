import os
import streamlit as st
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

load_dotenv()

# --- Setup clients (same as chat.py, adapted for the app) ---
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
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Source: page {chunk['page']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def answer_question(question):
    chunks = retrieve_chunks(question)
    context = build_context(chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "If the answer isn't in the context, say you don't have enough information — "
        "do not use outside knowledge. Always cite the page number(s) you used, like [page 2]."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = openai_client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content, chunks


# --- Streamlit UI ---
st.set_page_config(page_title="DocMind RAG", page_icon="📄")
st.title("📄 DocMind — Ask Your Document")
st.caption("RAG-powered Q&A over your PDF, built with Azure AI Search + Azure OpenAI")

# Keep chat history across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for chunk in msg["sources"]:
                    st.markdown(f"**Page {chunk['page']}:** {chunk['text'][:200]}...")

# Chat input
question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            answer, chunks = answer_question(question)
        st.write(answer)
        with st.expander("Sources"):
            for chunk in chunks:
                st.markdown(f"**Page {chunk['page']}:** {chunk['text'][:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": chunks})   