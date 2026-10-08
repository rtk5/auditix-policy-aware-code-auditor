"""Hybrid decision engine: CodeBERT as a fast gatekeeper, the LLM as the judge
for hard cases.

For each chunk we run the classifier once per retrieved policy and get a list of
probabilities ``p(violation)``. The routing rules are:

    any p in [0.35, 0.65]            -> AMBIGUOUS  : Kimi K2 decides, with the hint
    no ambiguous and any p > 0.65    -> VIOLATION  : fast path, classifier verdict
    no ambiguous and all p < 0.35    -> COMPLIANT  : fast path, classifier verdict

On the fast path the verdict is taken from the classifier. The LLM only writes
the explanation, unless ``explain_fast_path`` is False, in which case the LLM is
skipped entirely to save API calls.

The LLM call is passed in as a function, so the routing logic can be tested
without any network access.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from auditix.config import AMBIGUITY_HIGH, AMBIGUITY_LOW, CRITICAL_THRESHOLD

VALID_SEVERITIES = {"none", "low", "medium", "high", "critical"}

# Route names. Both fast-path routes share the same label, as in the notebook's decision_path.
ROUTE_AMBIGUOUS = "ambiguous→full-LLM"
ROUTE_VIOLATION = "high-confidence→LLM-explain"
ROUTE_COMPLIANT = "high-confidence→LLM-explain"

# Signature of the LLM callback: (chunk_name, code, summary, policies, hint) -> verdict dict.
LLMAudit = Callable[[str, str, str, Sequence[str], str], dict[str, Any]]


class Classifier(Protocol):
    """Anything with a ``violation_probability`` method (the real CodeBERT wrapper or a fake)."""

    def violation_probability(self, code: str, policy: str) -> float: ...


@dataclass(frozen=True)
class Routing:
    """The outcome of routing one chunk."""

    route: str            # one of ROUTE_* above
    max_probability: float  # highest p(violation) across the policies checked
    is_violation: bool    # verdict implied by the classifier (ignored when ambiguous)


def severity_from_probability(probability: float) -> str:
    """Map a classifier probability to a severity label (same bands as the notebook)."""
    if probability >= AMBIGUITY_HIGH:
        if probability >= CRITICAL_THRESHOLD:
            return "critical"
        if probability >= 0.80:
            return "high"
        return "medium"
    if probability >= AMBIGUITY_LOW:
        return "low"  # kept for completeness; the caller only passes p above AMBIGUITY_HIGH
    return "none"


def route_probabilities(probabilities: Sequence[float]) -> Routing:
    """Decide how a chunk should be handled from its per-policy probabilities."""
    if not probabilities:
        return Routing(ROUTE_COMPLIANT, 0.0, False)  # nothing to check: nothing to flag

    max_p = max(probabilities)
    # Ambiguity wins: if any policy is uncertain, the LLM decides for the whole chunk.
    if any(AMBIGUITY_LOW <= p <= AMBIGUITY_HIGH for p in probabilities):
        return Routing(ROUTE_AMBIGUOUS, max_p, max_p > AMBIGUITY_HIGH)
    if any(p > AMBIGUITY_HIGH for p in probabilities):
        return Routing(ROUTE_VIOLATION, max_p, True)
    return Routing(ROUTE_COMPLIANT, max_p, False)


def normalize_severity(value: Any, compliant: bool) -> str:
    """Return a valid severity label. Unknown values become "none" (compliant) or "unknown"."""
    severity = str(value or "").strip().lower()
    if severity in VALID_SEVERITIES:
        return severity
    return "none" if compliant else "unknown"


def audit_chunk_hybrid(
    chunk_name: str,
    code: str,
    summary: str,
    policies: Sequence[str],
    *,
    classifier: Classifier | None,
    llm_audit: LLMAudit,
    explain_fast_path: bool = True,
) -> dict[str, Any]:
    """Audit one chunk against its retrieved policies and return a verdict dict.

    Output keys: ``compliant``, ``violations``, ``explanation``, ``severity``,
    ``classifier_confidence`` (None in LLM-only mode) and ``decision_path``.
    """
    # ── LLM-only fallback (no classifier loaded) ─────────────────────────────
    if classifier is None:
        verdict = llm_audit(chunk_name, code, summary, policies, "")  # no hint without a classifier
        return _result(
            compliant=bool(verdict.get("compliant", False)),
            violations=list(verdict.get("violations", [])),
            explanation=str(verdict.get("explanation", "")),
            severity=normalize_severity(verdict.get("severity"), bool(verdict.get("compliant", False))),
            confidence=None,
            decision_path="[LLM-only]",
        )

    # ── Hybrid path ──────────────────────────────────────────────────────────
    # One CodeBERT score per retrieved policy, then decide the route.
    probabilities = [classifier.violation_probability(code, p) for p in policies]
    routing = route_probabilities(probabilities)
    confidence = round(routing.max_probability, 4)
    flagged = [p for p, vp in zip(policies, probabilities) if vp > AMBIGUITY_HIGH]

    if routing.route == ROUTE_AMBIGUOUS:
        # Uncertain: the LLM gives the verdict and is told the classifier was unsure.
        hint = (
            f"Fine-tuned CodeBERT: p(violation)={routing.max_probability:.1%}. "
            "The classifier is uncertain, so decide the verdict yourself and explain in detail."
        )
        verdict = llm_audit(chunk_name, code, summary, policies, hint)
        compliant = bool(verdict.get("compliant", False))  # missing verdict counts as non-compliant
        violations = list(verdict.get("violations", []))
        explanation = str(verdict.get("explanation", ""))
        severity = normalize_severity(verdict.get("severity"), compliant)
        decision_path = f"[Hybrid/{ROUTE_AMBIGUOUS}]"

    else:
        # Confident: the classifier's verdict is final. The LLM (optionally) only explains it.
        compliant = not routing.is_violation
        severity = severity_from_probability(routing.max_probability) if routing.is_violation else "none"
        decision_path = f"[Hybrid/{ROUTE_VIOLATION}]"
        verdict_word = "COMPLIANT" if compliant else "VIOLATION"

        if explain_fast_path:
            hint = (
                f"Fine-tuned CodeBERT: p(violation)={routing.max_probability:.1%}, "
                f"verdict={verdict_word}. The classifier is confident — "
                "explain the reasoning consistently with this verdict."
            )
            llm = llm_audit(chunk_name, code, summary, policies, hint)
            explanation = str(llm.get("explanation", ""))
            llm_violations = list(llm.get("violations", []))
        else:
            # Cost-saving mode: no API call, just a template explanation.
            explanation = (
                f"Verdict from the fine-tuned CodeBERT classifier alone ({verdict_word}). "
                "The LLM explanation was skipped (AUDITIX_EXPLAIN_FAST_PATH=0)."
            )
            llm_violations = []

        # Keep the LLM's list if it gave one; otherwise name the policies the classifier flagged.
        violations = [] if compliant else (llm_violations or [f"Violates policy: {p}" for p in flagged])

    return _result(
        compliant=compliant,
        violations=violations,
        explanation=f"Classifier p(violation)={routing.max_probability:.1%}. {explanation}".strip(),
        severity=severity,
        confidence=confidence,
        decision_path=decision_path,
    )


def _result(
    *,
    compliant: bool,
    violations: list[str],
    explanation: str,
    severity: str,
    confidence: float | None,
    decision_path: str,
) -> dict[str, Any]:
    """Build the verdict dict with a fixed set of keys (the schema every caller relies on)."""
    return {
        "compliant": compliant,
        "violations": violations,
        "explanation": explanation,
        "severity": severity,
        "classifier_confidence": confidence,
        "decision_path": decision_path,
    }
