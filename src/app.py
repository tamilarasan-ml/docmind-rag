
import streamlit as st
import os
import subprocess
import hashlib
import logging

from document_upload import validate_pdf_upload
from document_processor import process_uploaded_pdf
from document_embeddings import embed_document_chunks
from document_chat import answer_uploaded_document


# ---------------------------------------------------------
# Application diagnostics
# ---------------------------------------------------------

logger = logging.getLogger("docmind.diagnostics")


def log_app_exception(event, exc):
    """
    Log safe application error metadata without exposing
    raw exception details or sensitive information.
    """

    if os.getenv("DOCMIND_DIAGNOSTICS", "").lower() == "true":

        exception_hash = hashlib.sha256(
            str(exc).encode("utf-8")
        ).hexdigest()[:12]

        logger.error(
            "DOCMIND_APP_ERROR event=%s exception_type=%s exception_hash=%s",
            event,
            type(exc).__name__,
            exception_hash,
        )


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="DocMind RAG",
    page_icon="📄",
)

st.title("📄 DocMind — Ask Your Document")

st.caption(
    "RAG-powered document Q&A using Azure OpenAI "
    "and Azure AI Search."
)


# ---------------------------------------------------------
# Deployment information
# ---------------------------------------------------------

def get_deployment_commit():

    commit = os.getenv("GITHUB_SHA")

    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()

        except Exception:
            commit = "unavailable"

    return commit


deployment_commit = get_deployment_commit()

st.caption(
    f"Deployment Commit: `{deployment_commit}`"
)


# ---------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_messages" not in st.session_state:
    st.session_state.document_messages = []

if "uploaded_document_hash" not in st.session_state:
    st.session_state.uploaded_document_hash = None

if "uploaded_document_name" not in st.session_state:
    st.session_state.uploaded_document_name = None

if "embedded_document_chunks" not in st.session_state:
    st.session_state.embedded_document_chunks = None


# ---------------------------------------------------------
# Application mode selection
# ---------------------------------------------------------

# PDF upload remains disabled by default until security approval.
pdf_upload_enabled = (
    os.getenv("DOCMIND_ENABLE_PDF_UPLOAD", "").lower() == "true"
)

available_modes = ["Existing Knowledge Base"]

if pdf_upload_enabled:
    available_modes.append("Upload a PDF")

mode = st.radio(
    "Choose your document source",
    available_modes,
    horizontal=True,
)


# =========================================================
# MODE 1 — EXISTING KNOWLEDGE BASE
# =========================================================

if mode == "Existing Knowledge Base":

    st.subheader("Ask the existing knowledge base")

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

    question = st.chat_input(
        "Ask a question about the knowledge base..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching knowledge base and generating answer..."
            ):

                try:

                    # Lazy import to avoid initializing Azure Search
                    # when the user selects uploaded-document mode.

                    from chat import answer_question

                    answer, chunks = answer_question(question)

                    st.write(answer)

                    with st.expander("Sources"):

                        for chunk in chunks:

                            st.markdown(
                                f"**Page {chunk['page']}:** "
                                f"{chunk['text'][:200]}..."
                            )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": chunks,
                        }
                    )

                except Exception as exc:

                    log_app_exception(
                        "knowledge_base_answer",
                        exc,
                    )

                    st.error(
                        "Unable to generate an answer. "
                        "Please try again later."
                    )


# =========================================================
# MODE 2 — UPLOAD A PDF
# =========================================================

elif mode == "Upload a PDF":

    st.subheader("Upload your own PDF")

    st.caption(
        "Upload a text-based PDF and ask questions "
        "using document-specific retrieval."
    )

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        accept_multiple_files=False,
        key=f"document_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file is None:

        # Clear stale document state if the uploader is empty.

        if st.session_state.uploaded_document_hash is not None:

            st.session_state.embedded_document_chunks = None
            st.session_state.uploaded_document_hash = None
            st.session_state.uploaded_document_name = None
            st.session_state.document_messages = []

        st.info(
            "Upload a PDF to begin asking questions."
        )

    else:

        # -------------------------------------------------
        # Step 1: Validate the uploaded PDF
        # -------------------------------------------------

        is_valid, validation_message = validate_pdf_upload(
            uploaded_file
        )

        if not is_valid:

            # Invalidate the previously active document
            # when the user selects an invalid replacement.

            st.session_state.embedded_document_chunks = None
            st.session_state.uploaded_document_hash = None
            st.session_state.uploaded_document_name = None
            st.session_state.document_messages = []

            st.error(validation_message)

        else:

            pdf_bytes = uploaded_file.getvalue()

            current_document_hash = hashlib.sha256(
                pdf_bytes
            ).hexdigest()

            # -------------------------------------------------
            # Step 2: Process only when the document changes
            # -------------------------------------------------

            if (
                current_document_hash
                != st.session_state.uploaded_document_hash
            ):

                # Clear the previous document's session data.

                st.session_state.embedded_document_chunks = None
                st.session_state.document_messages = []

                st.session_state.uploaded_document_hash = None
                st.session_state.uploaded_document_name = None

                try:

                    with st.spinner(
                        "Processing PDF and generating embeddings..."
                    ):

                        # Extract text and create chunks.

                        chunks = process_uploaded_pdf(
                            pdf_bytes
                        )

                        # Generate document embeddings.

                        embedded_chunks = embed_document_chunks(
                            chunks
                        )

                    # Store the processed document in session state.

                    st.session_state.embedded_document_chunks = (
                        embedded_chunks
                    )

                    st.session_state.uploaded_document_hash = (
                        current_document_hash
                    )

                    st.session_state.uploaded_document_name = (
                        uploaded_file.name
                    )

                    st.success(
                        "PDF processed successfully."
                    )

                except Exception as exc:

                    log_app_exception(
                        "pdf_processing",
                        exc,
                    )

                    # Ensure failed processing does not leave
                    # stale document data in session state.

                    st.session_state.embedded_document_chunks = None
                    st.session_state.uploaded_document_hash = None
                    st.session_state.uploaded_document_name = None
                    st.session_state.document_messages = []

                    st.error(
                        "Unable to process this PDF. "
                        "Please try another document."
                    )

            # -------------------------------------------------
            # Step 3: Display document information
            # -------------------------------------------------

            if st.session_state.embedded_document_chunks:

                st.success(
                    f"Active document: "
                    f"{st.session_state.uploaded_document_name}"
                )

                st.caption(
                    f"Indexed chunks: "
                    f"{len(st.session_state.embedded_document_chunks)}"
                )

                if st.button(
                    "Remove Document",
                    type="secondary",
                    key="remove_uploaded_document",
                ):

                    st.session_state.embedded_document_chunks = None
                    st.session_state.uploaded_document_hash = None
                    st.session_state.uploaded_document_name = None
                    st.session_state.document_messages = []

                    st.session_state.uploader_key += 1

                    st.rerun()

                # -------------------------------------------------
                # Step 4: Display conversation history
                # -------------------------------------------------

                for msg in st.session_state.document_messages:

                    with st.chat_message(msg["role"]):

                        st.write(msg["content"])

                        if "sources" in msg:

                            with st.expander("Sources"):

                                for chunk in msg["sources"]:

                                    st.markdown(
                                        f"**Page {chunk['page']}:** "
                                        f"{chunk['text'][:200]}..."
                                    )

                # -------------------------------------------------
                # Step 5: Ask questions about the uploaded PDF
                # -------------------------------------------------

                question = st.chat_input(
                    "Ask a question about your uploaded PDF..."
                )

                if question:

                    st.session_state.document_messages.append(
                        {
                            "role": "user",
                            "content": question,
                        }
                    )

                    with st.chat_message("user"):
                        st.write(question)

                    with st.chat_message("assistant"):

                        with st.spinner(
                            "Retrieving relevant content and generating answer..."
                        ):

                            try:

                                answer, chunks = answer_uploaded_document(
                                    question=question,
                                    embedded_chunks=(
                                        st.session_state.embedded_document_chunks
                                    ),
                                )

                                st.write(answer)

                                with st.expander("Sources"):

                                    for chunk in chunks:

                                        st.markdown(
                                            f"**Page {chunk['page']}:** "
                                            f"{chunk['text'][:200]}..."
                                        )

                                st.session_state.document_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": answer,
                                        "sources": chunks,
                                    }
                                )

                            except Exception as exc:

                                log_app_exception(
                                    "uploaded_document_answer",
                                    exc,
                                )

                                st.error(
                                    "Unable to generate an answer "
                                    "from this document. Please try again."
                                )
