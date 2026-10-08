"""HTTP helper for the Groq API with rate-limit aware retries.

Groq's 429 errors say something like ``"Please try again in 2.45s."``. Instead of
sleeping a fixed time, we read that number and wait exactly that long (plus one
second of padding). This is the ``call_groq_with_retry()`` logic from the notebook.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

from auditix.config import GROQ_API_URL, GROQ_TIMEOUT_SECONDS, get_groq_api_key

logger = logging.getLogger(__name__)

DEFAULT_WAIT_SECONDS = 10.0
PADDING_SECONDS = 1.0
_WAIT_RE = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE)


def parse_wait_seconds(message: str, default: float = DEFAULT_WAIT_SECONDS) -> float:
    """Extract the retry delay from a Groq rate-limit message.

    >>> parse_wait_seconds("Rate limit reached. Please try again in 2.5s.")
    3.5
    """
    match = _WAIT_RE.search(message or "")
    if not match:
        return default + PADDING_SECONDS
    value = float(match.group(1))
    if match.group(2).lower() == "ms":
        value /= 1000.0
    return round(value + PADDING_SECONDS, 3)


def call_groq_with_retry(
    payload: dict[str, Any],
    max_retries: int = 5,
    *,
    api_key: str | None = None,
    sleep=time.sleep,
) -> dict[str, Any]:
    """POST a chat-completion request, retrying on rate limits.

    Returns the parsed JSON response. On failure after the retries (or on any
    non-rate-limit error) the error payload is returned, so callers can check for
    the ``"choices"`` key instead of catching exceptions.
    """
    key = api_key or get_groq_api_key()
    if not key:
        return {"error": {"message": "GROQ_API_KEY is not set"}}

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data: dict[str, Any] = {}
    for attempt in range(1, max_retries + 1):
        response = requests.post(
            GROQ_API_URL, headers=headers, json=payload, timeout=GROQ_TIMEOUT_SECONDS
        )
        try:
            data = response.json()
        except ValueError:
            return {"error": {"message": f"Non-JSON response (HTTP {response.status_code})"}}

        if "choices" in data:
            return data

        error = data.get("error", {}) or {}
        if error.get("code") != "rate_limit_exceeded":
            return data  # a real error: do not retry

        wait = parse_wait_seconds(error.get("message", ""))
        logger.warning("Groq rate limit; waiting %.1fs (attempt %d/%d)", wait, attempt, max_retries)
        sleep(wait)

    logger.error("Groq: max retries exceeded")
    return data if data else {}
