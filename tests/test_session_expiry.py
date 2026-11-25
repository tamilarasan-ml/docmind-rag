import unittest
from unittest.mock import patch

from session_expiry import (
    SESSION_TTL_SECONDS,
    mark_document_activity,
    expire_document_session,
)


class TestSessionExpiry(unittest.TestCase):

    def setUp(self):
        self.state = {
            "embedded_document_chunks": ["chunk-1"],
            "uploaded_document_hash": "abc123",
            "uploaded_document_name": "sample.pdf",
            "document_messages": [
                {"role": "user", "content": "Hello"}
            ],
            "messages": [
                {"role": "assistant", "content": "KB history"}
            ],
            "uploader_key": 0,
        }

    def test_does_not_expire_without_activity_timestamp(self):
        self.assertFalse(
            expire_document_session(self.state, now=5000)
        )

    def test_session_remains_active_before_timeout(self):
        mark_document_activity(self.state, now=1000)

        self.assertFalse(
            expire_document_session(
                self.state,
                now=1000 + SESSION_TTL_SECONDS - 1,
            )
        )
        self.assertEqual(
            self.state["uploaded_document_name"],
            "sample.pdf",
        )

    def test_session_expires_at_timeout(self):
        mark_document_activity(self.state, now=1000)

        self.assertTrue(
            expire_document_session(
                self.state,
                now=1000 + SESSION_TTL_SECONDS,
            )
        )

    def test_expiry_clears_uploaded_document_data(self):
        mark_document_activity(self.state, now=1000)

        expire_document_session(
            self.state,
            now=1000 + SESSION_TTL_SECONDS,
        )

        self.assertIsNone(self.state["embedded_document_chunks"])
        self.assertIsNone(self.state["uploaded_document_hash"])
        self.assertIsNone(self.state["uploaded_document_name"])
        self.assertEqual(self.state["document_messages"], [])
        self.assertIsNone(self.state["document_last_activity_at"])
        self.assertEqual(self.state["uploader_key"], 1)

    def test_expiry_preserves_knowledge_base_history(self):
        original_messages = list(self.state["messages"])
        mark_document_activity(self.state, now=1000)

        expire_document_session(
            self.state,
            now=1000 + SESSION_TTL_SECONDS,
        )

        self.assertEqual(self.state["messages"], original_messages)


    @patch("session_expiry.logger")
    def test_logs_when_activity_timestamp_is_missing(self, mock_logger):
        self.assertFalse(
            expire_document_session(self.state, now=5000)
        )

        mock_logger.info.assert_called_once_with(
            "DOCUMENT_SESSION_EXPIRY_TIMESTAMP_MISSING"
        )

    @patch("session_expiry.logger")
    def test_logs_elapsed_time_for_active_session(self, mock_logger):
        mark_document_activity(self.state, now=1000)

        self.assertFalse(
            expire_document_session(
                self.state,
                now=1000 + SESSION_TTL_SECONDS - 1,
            )
        )

        mock_logger.info.assert_called_once_with(
            "DOCUMENT_SESSION_EXPIRY_CHECK "
            "elapsed_seconds=%.2f threshold_seconds=%d",
            SESSION_TTL_SECONDS - 1,
            SESSION_TTL_SECONDS,
        )

    @patch("session_expiry.logger")
    def test_logs_when_session_expires(self, mock_logger):
        mark_document_activity(self.state, now=1000)

        self.assertTrue(
            expire_document_session(
                self.state,
                now=1000 + SESSION_TTL_SECONDS,
            )
        )

        self.assertEqual(mock_logger.info.call_count, 2)
        self.assertEqual(
            mock_logger.info.call_args_list[0].args[0],
            "DOCUMENT_SESSION_EXPIRY_CHECK "
            "elapsed_seconds=%.2f threshold_seconds=%d",
        )
        self.assertEqual(
            mock_logger.info.call_args_list[1].args[0],
            "DOCUMENT_SESSION_EXPIRY_TRIGGERED "
            "elapsed_seconds=%.2f",
        )



if __name__ == "__main__":
    unittest.main()
