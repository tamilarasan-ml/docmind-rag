
import streamlit as st

from chat import answer_question


st.set_page_config(page_title="DocMind RAG", page_icon="📄")
st.title("📄 DocMind — Ask Your Document")
st.caption(
    "RAG-powered Q&A over your PDF, built with Azure AI Search + Azure OpenAI"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if "sources" in msg:
            with st.expander("Sources"):
                for chunk in msg["sources"]:
                    st.markdown(
                        f"**Page {chunk['page']}:** "
                        f"{chunk['text'][:200]}..."
                    )

question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            answer, chunks = answer_question(question)

        st.write(answer)

        with st.expander("Sources"):
            for chunk in chunks:
                st.markdown(
                    f"**Page {chunk['page']}:** "
                    f"{chunk['text'][:200]}..."
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": chunks}
    )
