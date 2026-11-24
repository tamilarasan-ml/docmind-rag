import unittest

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


if __name__ == "__main__":
    unittest.main()
