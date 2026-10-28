"""Session-only authentication for the private, read-only research dashboard."""

from __future__ import annotations

import hashlib
import hmac
import math
import stat
import time
from pathlib import Path

import streamlit as st

AUTH_KEY = "_research_authenticated_digest"
INPUT_KEY = "_research_access_key"
ATTEMPTS_KEY = "_research_failed_attempts"
LOCK_KEY = "_research_locked_until"
ERROR_KEY = "_research_login_error"


def _clock() -> float:
    return time.time()


def read_token(path: Path | str) -> str:
    """Read an operator-owned secret without logging it or creating any files."""
    path = Path(path)
    properties = path.lstat()
    if not stat.S_ISREG(properties.st_mode) or properties.st_mode & 0o077:
        raise ValueError("A private regular dashboard token file is required")
    if properties.st_size > 2048:
        raise ValueError("Invalid dashboard access key")
    token = path.read_text().strip()
    if not 32 <= len(token) <= 1024 or "\n" in token or "\r" in token:
        raise ValueError("Invalid dashboard access key")
    return token


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def attempt_login(session, candidate: str, expected_digest: str, now: float) -> bool:
    """Validate one attempt; three failures lock this session for thirty seconds."""
    session[INPUT_KEY] = ""
    if now < session.get(LOCK_KEY, 0):
        session[ERROR_KEY] = "Please wait before trying again."
        return False
    candidate_digest = token_digest(candidate) if isinstance(candidate, str) else ""
    if hmac.compare_digest(candidate_digest, expected_digest):
        session[AUTH_KEY] = expected_digest
        session[ATTEMPTS_KEY] = 0
        session[LOCK_KEY] = 0
        session.pop(ERROR_KEY, None)
        return True
    session.pop(AUTH_KEY, None)
    failures = session.get(ATTEMPTS_KEY, 0) + 1
    session[ATTEMPTS_KEY] = failures
    if failures >= 3:
        session[LOCK_KEY] = now + 30
        session[ATTEMPTS_KEY] = 0
    session[ERROR_KEY] = "We could not verify that access key."
    return False


def _submit(expected_digest: str) -> None:
    attempt_login(
        st.session_state,
        st.session_state.get(INPUT_KEY, ""),
        expected_digest,
        _clock(),
    )


def _lock() -> None:
    for key in [AUTH_KEY, INPUT_KEY, ATTEMPTS_KEY, LOCK_KEY, ERROR_KEY]:
        st.session_state.pop(key, None)


def authentication_gate(token: str) -> bool:
    """Render only a generic login until the current secret has authenticated."""
    expected = token_digest(token)
    authenticated = st.session_state.get(AUTH_KEY)
    if isinstance(authenticated, str) and hmac.compare_digest(authenticated, expected):
        st.sidebar.button("Lock dashboard", on_click=_lock, key="lock_research_dashboard")
        return True
    # Rotation invalidates an existing session before private data are read.
    st.session_state.pop(AUTH_KEY, None)
    remaining = max(0, math.ceil(st.session_state.get(LOCK_KEY, 0) - _clock()))
    st.title("Private NFL research")
    st.caption("Enter the access key from your private host to view research and paper results.")
    if remaining:
        st.warning(f"Please wait {remaining} seconds before trying again.")
    elif st.session_state.get(ERROR_KEY):
        st.error("We could not verify that access key.")
    with st.form("research_dashboard_login"):
        st.text_input("Access key", type="password", key=INPUT_KEY, disabled=bool(remaining))
        st.form_submit_button(
            "Unlock dashboard", on_click=_submit, args=(expected,), disabled=bool(remaining)
        )
    return False
