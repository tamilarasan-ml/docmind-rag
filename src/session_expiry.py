"""
DOCMIND uploaded-document session expiry.

Tracks inactivity and clears uploaded-document state after
the configured session timeout.
"""
import logging
import time
from collections.abc import MutableMapping
from typing import Any

logger = logging.getLogger("docmind.diagnostics")

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

    If an uploaded document exists but its activity timestamp
    is missing, expire the document session defensively.

    Returns True when the session expires; otherwise False.
    """

    def clear_uploaded_document_session() -> None:
        session_state["embedded_document_chunks"] = None
        session_state["uploaded_document_hash"] = None
        session_state["uploaded_document_name"] = None
        session_state["document_messages"] = []
        session_state[ACTIVITY_KEY] = None
        session_state["uploader_key"] = (
            session_state.get("uploader_key", 0) + 1
        )

    last_activity = session_state.get(ACTIVITY_KEY)

    if last_activity is None:
        logger.info("DOCUMENT_SESSION_EXPIRY_TIMESTAMP_MISSING")

        has_uploaded_document = any(
            session_state.get(key)
            for key in (
                "embedded_document_chunks",
                "uploaded_document_hash",
                "uploaded_document_name",
            )
        )

        if not has_uploaded_document:
            return False

        logger.info(
            "DOCUMENT_SESSION_EXPIRY_TRIGGERED "
            "reason=missing_activity_timestamp"
        )
        clear_uploaded_document_session()
        return True

    current_time = time.time() if now is None else now
    elapsed_seconds = current_time - last_activity

    logger.info(
        "DOCUMENT_SESSION_EXPIRY_CHECK "
        "elapsed_seconds=%.2f threshold_seconds=%d",
        elapsed_seconds,
        SESSION_TTL_SECONDS,
    )

    if elapsed_seconds < SESSION_TTL_SECONDS:
        return False

    logger.info(
        "DOCUMENT_SESSION_EXPIRY_TRIGGERED "
        "elapsed_seconds=%.2f",
        elapsed_seconds,
    )

    clear_uploaded_document_session()
    return True
