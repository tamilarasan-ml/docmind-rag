import unittest
import fitz

from streamlit.testing.v1 import AppTest
from types import ModuleType
from unittest.mock import Mock, patch



class TestDocMindApp(unittest.TestCase):

    def setUp(self):
        self.upload_flag = patch.dict(
            "os.environ",
            {"DOCMIND_ENABLE_PDF_UPLOAD": "true"},
        )
        self.upload_flag.start()
        self.addCleanup(self.upload_flag.stop)

    def create_pdf(self, content):
        """Create a valid PDF in memory for AppTest."""
        pdf_document = fitz.open()

        pdf_document.new_page().insert_text(
            (72, 72),
            content,
        )

        pdf_bytes = pdf_document.tobytes()
        pdf_document.close()

        return pdf_bytes

    def create_chunks(self, chunk_id, content, embedding_value=0.1):
        """Create mock document chunks and embeddings."""
        chunks = [
            {
                "id": chunk_id,
                "text": content,
                "page": 1,
            }
        ]

        embedded_chunks = [
            {
                **chunks[0],
                "embedding": [embedding_value] * 1536,
            }
        ]

        return chunks, embedded_chunks

    def test_initializes_application_session_state(self):
        app = AppTest.from_file("src/app.py")
        app.run()

        self.assertEqual(app.session_state["messages"], [])
        self.assertEqual(app.session_state["document_messages"], [])

        self.assertIsNone(
            app.session_state["uploaded_document_hash"]
        )

        self.assertIsNone(
            app.session_state["uploaded_document_name"]
        )

        self.assertIsNone(
            app.session_state["embedded_document_chunks"]
        )

        self.assertEqual(
            app.radio[0].value,
            "Existing Knowledge Base",
        )

        self.assertEqual(len(app.exception), 0)

    @patch.dict(
        "os.environ",
        {"DOCMIND_ENABLE_PDF_UPLOAD": ""},
    )
    def test_pdf_upload_is_disabled_by_default(self):
        app = AppTest.from_file("src/app.py")
        app.run()

        self.assertEqual(
            app.radio[0].options,
            ["Existing Knowledge Base"],
        )

        self.assertEqual(len(app.file_uploader), 0)
        self.assertEqual(len(app.exception), 0)

    def test_displays_pdf_upload_mode(self):
        app = AppTest.from_file("src/app.py")
        app.run()

        app.radio[0].set_value("Upload a PDF").run()

        self.assertEqual(
            app.radio[0].value,
            "Upload a PDF",
        )

        self.assertEqual(len(app.file_uploader), 1)

        self.assertTrue(
            any(
                "Upload a PDF to begin asking questions."
                in element.value
                for element in app.info
            )
        )

        self.assertEqual(len(app.exception), 0)

    def test_rejects_malformed_pdf_upload(self):
        app = AppTest.from_file("src/app.py")
        app.run()

        app.radio[0].set_value("Upload a PDF").run()

        app.file_uploader[0].upload(
            "invalid.pdf",
            b"This is not a valid PDF document.",
            "application/pdf",
        ).run()

        self.assertTrue(len(app.error) > 0)

        self.assertIsNone(
            app.session_state["embedded_document_chunks"]
        )

        self.assertEqual(len(app.exception), 0)

    @patch("document_embeddings.embed_document_chunks")
    @patch("document_processor.process_uploaded_pdf")
    def test_processes_valid_pdf_upload(
        self,
        mock_process,
        mock_embed,
    ):
        app = AppTest.from_file("src/app.py")
        app.run()

        app.radio[0].set_value("Upload a PDF").run()

        pdf_bytes = self.create_pdf(
            "DOCMIND payment processing test document."
        )

        chunks, embedded_chunks = self.create_chunks(
            "chunk_0",
            "Payment processing information.",
        )

        mock_process.return_value = chunks
        mock_embed.return_value = embedded_chunks

        app.file_uploader[0].upload(
            "sample.pdf",
            pdf_bytes,
            "application/pdf",
        ).run()

        self.assertEqual(
            app.session_state["uploaded_document_name"],
            "sample.pdf",
        )

        self.assertIsNotNone(
            app.session_state["uploaded_document_hash"]
        )

        self.assertEqual(
            app.session_state["embedded_document_chunks"],
            embedded_chunks,
        )

        mock_process.assert_called_once_with(pdf_bytes)
        mock_embed.assert_called_once_with(chunks)

        self.assertEqual(len(app.exception), 0)

    @patch("document_embeddings.embed_document_chunks")
    @patch("document_processor.process_uploaded_pdf")
    def test_replaces_existing_uploaded_pdf(
        self,
        mock_process,
        mock_embed,
    ):
        app = AppTest.from_file("src/app.py")
        app.run()

        app.radio[0].set_value("Upload a PDF").run()

        # Create two different PDFs.
        first_pdf_bytes = self.create_pdf(
            "First DOCMIND test document."
        )

        second_pdf_bytes = self.create_pdf(
            "Second DOCMIND replacement document."
        )

        first_chunks, first_embedded = self.create_chunks(
            "first_chunk",
            "First document content.",
            0.1,
        )

        second_chunks, second_embedded = self.create_chunks(
            "second_chunk",
            "Second document content.",
            0.2,
        )

        mock_process.side_effect = [
            first_chunks,
            second_chunks,
        ]

        mock_embed.side_effect = [
            first_embedded,
            second_embedded,
        ]

        # Upload the first PDF.
        app.file_uploader[0].upload(
            "first.pdf",
            first_pdf_bytes,
            "application/pdf",
        ).run()

        first_hash = app.session_state["uploaded_document_hash"]

        self.assertEqual(
            app.session_state["uploaded_document_name"],
            "first.pdf",
        )

        self.assertEqual(
            app.session_state["embedded_document_chunks"],
            first_embedded,
        )

        # Simulate existing conversation history.
        app.session_state["document_messages"] = [
            {
                "role": "user",
                "content": "Previous document question.",
            }
        ]

        # Upload a different PDF.
        app.file_uploader[0].upload(
            "second.pdf",
            second_pdf_bytes,
            "application/pdf",
        ).run()

        self.assertEqual(
            app.session_state["uploaded_document_name"],
            "second.pdf",
        )

        self.assertNotEqual(
            app.session_state["uploaded_document_hash"],
            first_hash,
        )

        self.assertEqual(
            app.session_state["embedded_document_chunks"],
            second_embedded,
        )

        # Previous conversation must be cleared.
        self.assertEqual(
            app.session_state["document_messages"],
            [],
        )

        self.assertEqual(mock_process.call_count, 2)
        self.assertEqual(mock_embed.call_count, 2)

        self.assertEqual(len(app.exception), 0)

    @patch("document_embeddings.embed_document_chunks")
    @patch("document_processor.process_uploaded_pdf")
    def test_removes_uploaded_document(
        self,
        mock_process,
        mock_embed,
    ):
        app = AppTest.from_file("src/app.py")
        app.run()

        app.radio[0].set_value("Upload a PDF").run()

        pdf_bytes = self.create_pdf(
            "DOCMIND document removal test."
        )

        chunks, embedded_chunks = self.create_chunks(
            "chunk_0",
            "Document removal test content.",
        )

        mock_process.return_value = chunks
        mock_embed.return_value = embedded_chunks

        # Upload the PDF.
        app.file_uploader[0].upload(
            "remove_test.pdf",
            pdf_bytes,
            "application/pdf",
        ).run()

        self.assertEqual(
            app.session_state["uploaded_document_name"],
            "remove_test.pdf",
        )

        # Add conversation history to verify cleanup.
        app.session_state["document_messages"] = [
            {
                "role": "user",
                "content": "Test question.",
            }
        ]

        previous_uploader_key = app.session_state["uploader_key"]

        # Click Remove Document.
        app.button[0].click().run()

        # Verify document state is cleared.
        self.assertIsNone(
            app.session_state["embedded_document_chunks"]
        )

        self.assertIsNone(
            app.session_state["uploaded_document_hash"]
        )

        self.assertIsNone(
            app.session_state["uploaded_document_name"]
        )

        self.assertEqual(
            app.session_state["document_messages"],
            [],
        )

        # Verify uploader is reset.
        self.assertEqual(
            app.session_state["uploader_key"],
            previous_uploader_key + 1,
        )

        self.assertEqual(len(app.exception), 0)

    @patch("document_chat.answer_uploaded_document")
    @patch("document_embeddings.embed_document_chunks")
    @patch("document_processor.process_uploaded_pdf")
    def test_answers_question_about_uploaded_pdf(
        self,
        mock_process,
        mock_embed,
        mock_answer,
    ):
        app = AppTest.from_file("src/app.py")
        app.run(timeout=10)

        app.radio[0].set_value("Upload a PDF").run(timeout=10)

        pdf_bytes = self.create_pdf(
            "DOCMIND payment processing test document."
        )

        chunks, embedded_chunks = self.create_chunks(
            "chunk_0",
            "Payment processing information.",
        )

        expected_answer = (
            "Payment processing is handled through the payment platform."
        )

        mock_process.return_value = chunks
        mock_embed.return_value = embedded_chunks
        mock_answer.return_value = (
            expected_answer,
            chunks,
        )

        # Upload the PDF.
        app.file_uploader[0].upload(
            "qa_test.pdf",
            pdf_bytes,
            "application/pdf",
        ).run(timeout=10)

        # Submit a question.
        app.chat_input[0].set_value(
            "How does payment processing work?"
        ).run(timeout=10)

        # Verify the answer function invocation.
        mock_answer.assert_called_once_with(
            question="How does payment processing work?",
            embedded_chunks=embedded_chunks,
        )

        # Verify conversation history.
        messages = app.session_state["document_messages"]

        self.assertEqual(len(messages), 2)

        self.assertEqual(
            messages[0]["role"],
            "user",
        )

        self.assertEqual(
            messages[0]["content"],
            "How does payment processing work?",
        )

        self.assertEqual(
            messages[1]["role"],
            "assistant",
        )

        self.assertEqual(
            messages[1]["content"],
            expected_answer,
        )

        self.assertEqual(
            messages[1]["sources"],
            chunks,
        )

        self.assertEqual(len(app.exception), 0)

    @patch("document_embeddings.embed_document_chunks")
    @patch("document_processor.process_uploaded_pdf")
    def test_clearing_uploader_clears_document_state(
        self,
        mock_process,
        mock_embed,
    ):
        app = AppTest.from_file("src/app.py")
        app.run()

        app.radio[0].set_value("Upload a PDF").run()

        pdf_bytes = self.create_pdf(
            "DOCMIND uploader cleanup test."
        )

        chunks, embedded_chunks = self.create_chunks(
            "cleanup_chunk",
            "Uploader cleanup test content.",
        )

        mock_process.return_value = chunks
        mock_embed.return_value = embedded_chunks

        # Upload a PDF.
        app.file_uploader[0].upload(
            "cleanup_test.pdf",
            pdf_bytes,
            "application/pdf",
        ).run()

        self.assertIsNotNone(
            app.session_state["uploaded_document_hash"]
        )

        # Seed conversation history.
        app.session_state["document_messages"] = [
            {
                "role": "user",
                "content": "Previous question.",
            }
        ]

        # Clear the uploader.
        app.file_uploader[0].set_value(None).run()

        # Verify stale document state is cleared.
        self.assertIsNone(
            app.session_state["embedded_document_chunks"]
        )

        self.assertIsNone(
            app.session_state["uploaded_document_hash"]
        )

        self.assertIsNone(
            app.session_state["uploaded_document_name"]
        )

        self.assertEqual(
            app.session_state["document_messages"],
            [],
        )

        self.assertEqual(len(app.exception), 0)

    @patch.dict("os.environ", {"DOCMIND_DIAGNOSTICS": "true"})
    @patch("document_embeddings.embed_document_chunks")
    @patch("document_processor.process_uploaded_pdf")
    def test_handles_pdf_processing_failure(
        self,
        mock_process,
        mock_embed,
    ):
        app = AppTest.from_file("src/app.py")
        app.run()

        app.radio[0].set_value("Upload a PDF").run()

        pdf_bytes = self.create_pdf(
            "DOCMIND PDF processing failure test."
        )

        mock_process.side_effect = RuntimeError(
            "Simulated PDF processing failure."
        )

        # Capture application diagnostic logging.
        with self.assertLogs(
            "docmind.diagnostics",
            level="ERROR",
        ) as captured_logs:

            app.file_uploader[0].upload(
                "failure_test.pdf",
                pdf_bytes,
                "application/pdf",
            ).run()

        # Verify the generic processing error is displayed.
        self.assertTrue(
            any(
                "Unable to process this PDF"
                in element.value
                for element in app.error
            )
        )

        # Verify raw exception details are not exposed to users.
        self.assertFalse(
            any(
                "Simulated PDF processing failure"
                in element.value
                for element in app.error
            )
        )

        # Verify diagnostic event and exception type are logged.
        self.assertTrue(
            any(
                "DOCMIND_APP_ERROR" in message
                and "event=pdf_processing" in message
                and "exception_type=RuntimeError" in message
                for message in captured_logs.output
            )
        )

        # Verify raw exception details are not logged.
        self.assertFalse(
            any(
                "Simulated PDF processing failure"
                in message
                for message in captured_logs.output
            )
        )

        # Verify no document state remains after failure.
        self.assertIsNone(
            app.session_state["embedded_document_chunks"]
        )

        self.assertIsNone(
            app.session_state["uploaded_document_hash"]
        )

        self.assertIsNone(
            app.session_state["uploaded_document_name"]
        )

        self.assertEqual(
            app.session_state["document_messages"],
            [],
        )

        # Embedding should not run when processing fails.
        mock_embed.assert_not_called()

        self.assertEqual(len(app.exception), 0)

    def test_answers_question_using_existing_knowledge_base(self):

        # Create a mock chat module to avoid Azure connections.
        mock_chat = ModuleType("chat")

        mock_chat.answer_question = Mock(
            return_value=(
                "Payment exceptions are investigated and resolved.",
                [
                    {
                        "id": "kb_chunk_1",
                        "text": "Payment exception resolution process.",
                        "page": 5,
                    }
                ],
            )
        )

        with patch.dict("sys.modules", {"chat": mock_chat}):

            app = AppTest.from_file("src/app.py")
            app.run()

            # Confirm the default mode is Knowledge Base.
            self.assertEqual(
                app.radio[0].value,
                "Existing Knowledge Base",
            )

            # Seed previous conversation history.
            app.session_state["messages"] = [
                {
                    "role": "user",
                    "content": "What is a payment exception?",
                },
                {
                    "role": "assistant",
                    "content": "It is a payment that requires investigation.",
                    "sources": [
                        {
                            "id": "previous_chunk",
                            "text": "Payment exception definition.",
                            "page": 3,
                        }
                    ],
                },
            ]

            # Submit a new question.
            app.chat_input[0].set_value(
                "How are payment exceptions resolved?"
            ).run()

            # Verify the Knowledge Base function was called.
            mock_chat.answer_question.assert_called_once_with(
                "How are payment exceptions resolved?"
            )

            # Verify the new conversation history.
            messages = app.session_state["messages"]

            self.assertEqual(len(messages), 4)

            self.assertEqual(
                messages[2]["role"],
                "user",
            )

            self.assertEqual(
                messages[2]["content"],
                "How are payment exceptions resolved?",
            )

            self.assertEqual(
                messages[3]["role"],
                "assistant",
            )

            self.assertEqual(
                messages[3]["content"],
                "Payment exceptions are investigated and resolved.",
            )

            self.assertEqual(
                messages[3]["sources"][0]["page"],
                5,
            )

            self.assertEqual(len(app.exception), 0)

    def test_handles_existing_knowledge_base_answer_failure(self):

        # Mock the chat module to prevent real Azure connections.
        mock_chat = ModuleType("chat")

        mock_chat.answer_question = Mock(
            side_effect=RuntimeError(
                "Simulated Knowledge Base failure."
            )
        )

        with patch.dict("sys.modules", {"chat": mock_chat}):

            app = AppTest.from_file("src/app.py")
            app.run()

            question = "How are payment exceptions resolved?"

            app.chat_input[0].set_value(question).run()

            # Verify the Knowledge Base function was called.
            mock_chat.answer_question.assert_called_once_with(
                question
            )

            # Verify the error is displayed.
            self.assertTrue(
                any(
                    "Unable to generate an answer"
                    in element.value
                    for element in app.error
                )
            )

            # The user question should remain in history.
            messages = app.session_state["messages"]

            self.assertEqual(len(messages), 1)

            self.assertEqual(
                messages[0]["role"],
                "user",
            )

            self.assertEqual(
                messages[0]["content"],
                question,
            )

            # No assistant answer should be appended.
            self.assertEqual(len(app.exception), 0)

    @patch("document_chat.answer_uploaded_document")
    @patch("document_embeddings.embed_document_chunks")
    @patch("document_processor.process_uploaded_pdf")
    def test_handles_uploaded_pdf_answer_failure(
        self,
        mock_process,
        mock_embed,
        mock_answer,
    ):
        app = AppTest.from_file("src/app.py")
        app.run()

        app.radio[0].set_value("Upload a PDF").run()

        pdf_bytes = self.create_pdf(
            "DOCMIND PDF answer failure test."
        )

        chunks, embedded_chunks = self.create_chunks(
            "failure_chunk",
            "Payment processing test content.",
        )

        mock_process.return_value = chunks
        mock_embed.return_value = embedded_chunks

        mock_answer.side_effect = RuntimeError(
            "Simulated PDF answer-generation failure."
        )

        # Upload the PDF.
        app.file_uploader[0].upload(
            "answer_failure.pdf",
            pdf_bytes,
            "application/pdf",
        ).run()

        question = "How does payment processing work?"

        # Submit a question.
        app.chat_input[0].set_value(question).run()

        # Verify the answer function was called.
        mock_answer.assert_called_once_with(
            question=question,
            embedded_chunks=embedded_chunks,
        )

        # Verify the error is displayed.
        self.assertTrue(
            any(
                "Unable to generate an answer"
                in element.value
                for element in app.error
            )
        )

        # User question should remain in history.
        messages = app.session_state["document_messages"]

        self.assertEqual(len(messages), 1)

        self.assertEqual(
            messages[0]["role"],
            "user",
        )

        self.assertEqual(
            messages[0]["content"],
            question,
        )

        self.assertEqual(len(app.exception), 0)

    @patch("document_embeddings.embed_document_chunks")
    @patch("document_processor.process_uploaded_pdf")
    def test_displays_uploaded_pdf_conversation_history(
        self,
        mock_process,
        mock_embed,
    ):
        app = AppTest.from_file("src/app.py")
        app.run()

        app.radio[0].set_value("Upload a PDF").run()

        pdf_bytes = self.create_pdf(
            "DOCMIND conversation history test."
        )

        chunks, embedded_chunks = self.create_chunks(
            "history_chunk",
            "Payment processing information.",
        )

        mock_process.return_value = chunks
        mock_embed.return_value = embedded_chunks

        app.file_uploader[0].upload(
            "history_test.pdf",
            pdf_bytes,
            "application/pdf",
        ).run()

        # Seed an existing conversation.
        app.session_state["document_messages"] = [
            {
                "role": "user",
                "content": "How does payment processing work?",
            },
            {
                "role": "assistant",
                "content": "Payment processing follows defined transaction stages.",
                "sources": [
                    {
                        "id": "history_source",
                        "text": "Payment processing transaction stages.",
                        "page": 4,
                    }
                ],
            },
        ]

        # Rerun the application to render conversation history.
        app.run()

        self.assertEqual(
            len(app.session_state["document_messages"]),
            2,
        )

        self.assertTrue(
            any(
                "How does payment processing work?"
                in element.value
                for element in app.markdown
            )
        )

        self.assertTrue(
            any(
                "Payment processing follows defined transaction stages."
                in element.value
                for element in app.markdown
            )
        )

        self.assertTrue(
            any(
                "Page 4"
                in element.value
                for element in app.markdown
            )
        )

        self.assertEqual(len(app.exception), 0)

    def test_deployment_commit_fallback_when_git_lookup_fails(self):

        with patch.dict("os.environ", {"GITHUB_SHA": ""}):
            with patch(
                "subprocess.check_output",
                side_effect=RuntimeError(
                    "Simulated Git lookup failure."
                ),
            ):
                app = AppTest.from_file("src/app.py")
                app.run()

        self.assertTrue(
            any(
                "unavailable" in element.value
                for element in app.caption
            )
        )

        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()