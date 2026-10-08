"""Two LLM roles, both served by Groq:

* ``summarize_chunk`` — LLaMA 3.1 8B writes a 2–3 sentence business summary. The
  summary is only used as a *search query* for policy retrieval, so it is cheap.
* ``audit_chunk`` — Kimi K2 returns a structured JSON verdict
  ``{compliant, violations, explanation, severity}``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from auditix.config import GROQ_AUDIT_MODEL, GROQ_SUMMARY_MODEL
from auditix.llm.retry import call_groq_with_retry

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = (
    "You are a code analyst. Describe what this code does in 2-3 sentences, "
    "focusing on business logic, data access, and external calls."
)

AUDIT_SYSTEM_PROMPT = (
    "You are a policy compliance auditor. Always respond with valid JSON only. "
    "No markdown, no text outside the JSON object."
)

VALID_SEVERITIES = {"none", "low", "medium", "high", "critical"}


def summarize_chunk(code: str) -> str:
    """Return a short business-level summary of ``code``."""
    payload = {
        "model": GROQ_SUMMARY_MODEL,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"```python\n{code}\n```"},
        ],
        "max_tokens": 150,
    }
    data = call_groq_with_retry(payload)
    if "choices" not in data:
        message = (data.get("error") or {}).get("message", str(data))
        return f"Summary unavailable: {message}"
    return data["choices"][0]["message"]["content"].strip()


def build_audit_prompt(
    chunk_name: str,
    code: str,
    summary: str,
    policies: Sequence[str],
    classifier_hint: str = "",
) -> str:
    policy_text = "\n".join(f"- {p}" for p in policies)
    hint_block = f"\nPRE-CLASSIFICATION HINT (fine-tuned CodeBERT): {classifier_hint}\n" if classifier_hint else ""
    return (
        "You are a compliance auditor. Analyze the following code against the provided policies.\n\n"
        f"CODE CHUNK: {chunk_name}\n```python\n{code}\n```\n\n"
        f"CODE SUMMARY: {summary}\n"
        f"{hint_block}\n"
        f"RELEVANT POLICIES:\n{policy_text}\n\n"
        "Respond in JSON with this exact format:\n"
        '{"compliant": true or false, '
        '"violations": ["list of specific violations found, empty if none"], '
        '"explanation": "clear explanation of your reasoning — state which policy is violated and why, '
        'or why the code is compliant", '
        '"severity": "none | low | medium | high | critical"}'
    )


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Parse the model's reply, tolerating ```json fences and stray text.

    Falls back to a non-compliant result with the raw text if nothing parses,
    which is the same conservative behaviour as the notebook.
    """
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"compliant": False, "violations": ["JSON parse error"], "explanation": raw, "severity": "unknown"}


def audit_chunk(
    chunk_name: str,
    code: str,
    summary: str,
    policies: Sequence[str],
    classifier_hint: str = "",
) -> dict[str, Any]:
    """Ask Kimi K2 for a structured compliance verdict on one chunk."""
    payload = {
        "model": GROQ_AUDIT_MODEL,
        "messages": [
            {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": build_audit_prompt(chunk_name, code, summary, policies, classifier_hint)},
        ],
        "max_tokens": 500,
    }
    data = call_groq_with_retry(payload)
    if "choices" not in data:
        return {
            "compliant": False,
            "violations": ["API error"],
            "explanation": str(data),
            "severity": "unknown",
        }
    return parse_llm_json(data["choices"][0]["message"]["content"].strip())
