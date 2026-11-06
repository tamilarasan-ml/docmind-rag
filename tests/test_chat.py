import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestChat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env_patcher = patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_API_VERSION": "2024-02-01",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "test-embedding",
                "AZURE_SEARCH_QUERY_KEY": "test-search-key",
                "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net",
                "AZURE_SEARCH_INDEX_NAME": "test-index",
            },
        )
        cls.env_patcher.start()

        cls.azure_patcher = patch.dict(
            sys.modules,
            {
                "azure": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.search": MagicMock(),
                "azure.search.documents": MagicMock(),
                "azure.search.documents.models": MagicMock(),
            },
        )
        cls.azure_patcher.start()

        cls.openai_patcher = patch(
            "openai.AzureOpenAI",
            create=True,
        )
        cls.openai_patcher.start()

        sys.modules.pop("chat", None)
        cls.chat = importlib.import_module("chat")

    @classmethod
    def tearDownClass(cls):
        cls.openai_patcher.stop()
        cls.azure_patcher.stop()
        cls.env_patcher.stop()
        sys.modules.pop("chat", None)

    def setUp(self):
        self.question = "What is JQL?"

        self.chunks = [
            {
                "id": "chunk_1",
                "text": "JQL is used to search Jira issues.",
                "page": 2,
            }
        ]

        # Preserve the original module-level clients.
        self.original_openai_client = self.chat.openai_client
        self.original_search_client = self.chat.search_client

        # Reset shared mock state between tests.
        if hasattr(self.chat.search_client, "reset_mock"):
            self.chat.search_client.reset_mock()

        self.chat.search_client.search.return_value = []

    def tearDown(self):
        self.chat.openai_client = self.original_openai_client
        self.chat.search_client = self.original_search_client

    def test_rejects_empty_retrieval_question(self):
        for question in ["", "   ", None]:
            with self.subTest(question=question):
                with self.assertRaisesRegex(
                    ValueError,
                    "The question cannot be empty.",
                ):
                    self.chat.retrieve_chunks(question)

    @patch("chat.select_retrieval_query")
    def test_retrieves_chunks_using_vector_search(self, mock_routing):
        mock_routing.return_value = self.question

        embedding_client = MagicMock()
        embedding_client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1] * 1536)
            ]
        )

        self.chat.openai_client = embedding_client

        search_results = [
            {
                "id": "chunk_1",
                "text": "JQL is used to search Jira issues.",
                "page": 2,
            }
        ]

        self.chat.search_client.search.return_value = search_results

        result = self.chat.retrieve_chunks(self.question)

        self.assertEqual(result, search_results)
        embedding_client.embeddings.create.assert_called_once_with(
            model="test-embedding",
            input=self.question,
        )

        search_call = self.chat.search_client.search.call_args

        self.assertIsNone(search_call.kwargs["search_text"])
        self.assertEqual(
            search_call.kwargs["select"],
            ["id", "text", "page"],
        )

        vector_query = self.chat.VectorizedQuery.call_args.kwargs

        self.assertEqual(
            vector_query["vector"],
            [0.1] * 1536,
        )
        self.assertEqual(
            vector_query["k_nearest_neighbors"],
            3,
        )
        self.assertEqual(vector_query["fields"], "embedding")
        mock_routing.assert_called_once_with(self.question)
        embedding_client.embeddings.create.assert_called_once()
        self.chat.search_client.search.assert_called_once()

    @patch("chat.select_retrieval_query")
    def test_uses_routed_query_for_embedding(self, mock_routing):
        routed_query = "JQL summary negative matching"
        mock_routing.return_value = routed_query

        embedding_client = MagicMock()
        embedding_client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1] * 1536)
            ]
        )

        self.chat.openai_client = embedding_client
        self.chat.search_client.search.return_value = []

        self.chat.retrieve_chunks(
            "How can I find issues whose summary does not contain the word Task?"
        )

        self.assertEqual(
            embedding_client.embeddings.create.call_args.kwargs["input"],
            routed_query,
        )

    def test_rejects_empty_answer_question(self):
        for question in ["", "   ", None]:
            with self.subTest(question=question):
                with self.assertRaisesRegex(
                    ValueError,
                    "The question cannot be empty.",
                ):
                    self.chat.answer_question(question)

    @patch("chat.answer_with_chunks")
    @patch("chat.retrieve_chunks")
    def test_answers_question_using_retrieved_chunks(
        self,
        mock_retrieve,
        mock_answer,
    ):
        expected = ("JQL is used to search issues.", self.chunks)

        mock_retrieve.return_value = self.chunks
        mock_answer.return_value = expected

        result = self.chat.answer_question(self.question)

        self.assertEqual(result, expected)
        mock_retrieve.assert_called_once_with(self.question)
        mock_answer.assert_called_once_with(
            self.question,
            self.chunks,
            answer_transform=unittest.mock.ANY,
        )

    @patch("chat.answer_with_chunks")
    @patch("chat.retrieve_chunks")
    def test_applies_targeted_jql_correction(
        self,
        mock_retrieve,
        mock_answer,
    ):
        question = (
            "How can I find issues whose summary does not contain "
            "the word Task?"
        )

        mock_retrieve.return_value = self.chunks

        transform = mock_answer.call_args

        def capture_transform():
            return transform

        mock_answer.return_value = ("corrected", self.chunks)

        self.chat.answer_question(question)

        answer_transform = mock_answer.call_args.kwargs["answer_transform"]

        original = 'summary !~ "Task"'
        incorrect = '!~ summary !~ "Task"'

        self.assertEqual(
            answer_transform(incorrect),
            original,
        )

    @patch("chat.answer_with_chunks")
    @patch("chat.retrieve_chunks")
    def test_preserves_answer_for_unrelated_question(
        self,
        mock_retrieve,
        mock_answer,
    ):
        mock_retrieve.return_value = self.chunks
        mock_answer.return_value = ("Original answer", self.chunks)

        self.chat.answer_question("What is JQL?")

        answer_transform = mock_answer.call_args.kwargs["answer_transform"]

        answer = "JQL is a query language."

        self.assertEqual(answer_transform(answer), answer)

    def test_logs_diagnostics_when_enabled(self):
        with patch.object(self.chat, "DIAGNOSTICS_ENABLED", True):
            with patch.object(self.chat.logger, "warning") as mock_warning:
                self.chat.diagnostic_log(
                    "test_event",
                    question_hash="abc123",
                )

        mock_warning.assert_called_once()

        log_args = mock_warning.call_args.args

        self.assertEqual(log_args[0], "DOCMIND_DIAG %s %s")
        self.assertEqual(log_args[1], "test_event")
        self.assertIn("abc123", log_args[2])

    def test_rejects_missing_search_query_credential(self):
        with patch.dict(
            os.environ,
            {"AZURE_SEARCH_QUERY_KEY": ""},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Azure AI Search query credential is not configured.",
            ):
                importlib.reload(self.chat)

        # Restore the module to its normal configured state.
        importlib.reload(self.chat)


if __name__ == "__main__":
    unittest.main()
