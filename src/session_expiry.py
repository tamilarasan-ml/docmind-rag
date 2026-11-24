"""
DOCMIND uploaded-document session expiry.

Tracks inactivity and clears uploaded-document state after
the configured session timeout.
"""

import time
from collections.abc import MutableMapping
from typing import Any


SESSION_TTL_SECONDS = 30 * 60

ACTIVITY_KEY = "document_last_activity_at"


def mark_document_activity(
    session_state: MutableMapping[str, Any],
    now: float | None = None,
) -> None:
    """Record the latest meaningful document-session activity."""

    session_state[ACTIVITY_KEY] = (
        time.time() if now is None else now
    )


def expire_document_session(
    session_state: MutableMapping[str, Any],
    now: float | None = None,
) -> bool:
    """
    Expire an uploaded-document session after 30 minutes
    of inactivity.

    Returns True when the session expires; otherwise False.
    """

    last_activity = session_state.get(ACTIVITY_KEY)

    if last_activity is None:
        return False

    current_time = time.time() if now is None else now

    if current_time - last_activity < SESSION_TTL_SECONDS:
        return False

    # Clear all uploaded-document data and conversation history.
    session_state["embedded_document_chunks"] = None
    session_state["uploaded_document_hash"] = None
    session_state["uploaded_document_name"] = None
    session_state["document_messages"] = []

    # Reset activity tracking.
    session_state[ACTIVITY_KEY] = None

    # Reset the uploader so the old PDF cannot be reused.
    session_state["uploader_key"] = (
        session_state.get("uploader_key", 0) + 1
    )

    return True
