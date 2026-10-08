import pytest

from auditix.hybrid_engine import (
    ROUTE_AMBIGUOUS,
    ROUTE_VIOLATION,
    audit_chunk_hybrid,
    route_probabilities,
    severity_from_probability,
)

POLICIES = ["policy A", "policy B", "policy C"]


# Stub classifier returning fixed per-policy probabilities, so routing is predictable.
class FakeClassifier:
    """Returns a fixed probability per policy and records the calls."""

    def __init__(self, probs):
        self.probs = dict(zip(POLICIES, probs))
        self.calls = 0

    def violation_probability(self, code, policy):
        self.calls += 1
        return self.probs[policy]


# Stub LLM that records each call and returns a canned verdict.
class FakeLLM:
    def __init__(self, verdict=None):
        self.verdict = verdict or {
            "compliant": False,
            "violations": ["LLM says: bad"],
            "explanation": "LLM explanation",
            "severity": "high",
        }
        self.calls = []

    def __call__(self, name, code, summary, policies, hint):
        self.calls.append({"name": name, "hint": hint, "policies": list(policies)})
        return dict(self.verdict)


# ── Routing rules ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "probs, route",
    [
        ([0.10, 0.20, 0.05], "high-confidence→LLM-explain"),   # all low -> compliant fast path
        ([0.90, 0.10, 0.05], ROUTE_VIOLATION),                  # one high -> violation fast path
        ([0.50, 0.10, 0.05], ROUTE_AMBIGUOUS),                  # one in the 35–65% band
        ([0.35, 0.99, 0.05], ROUTE_AMBIGUOUS),                  # boundary 0.35 is ambiguous
        ([0.65, 0.05, 0.05], ROUTE_AMBIGUOUS),                  # boundary 0.65 is ambiguous
    ],
)
# Each probability pattern goes to the expected route, including both threshold boundaries.
def test_route_probabilities(probs, route):
    assert route_probabilities(probs).route == route


# With no policies there is nothing to flag.
def test_route_with_no_policies_is_compliant():
    routing = route_probabilities([])
    assert routing.is_violation is False and routing.max_probability == 0.0


@pytest.mark.parametrize(
    "p, severity",
    [(0.97, "critical"), (0.85, "high"), (0.70, "medium"), (0.50, "low"), (0.10, "none")],
)
# Each probability range maps to its severity label.
def test_severity_bands(p, severity):
    assert severity_from_probability(p) == severity


# ── Engine behaviour ─────────────────────────────────────────────────────────

# On the fast path the classifier decides, even when the LLM disagrees.
def test_fast_path_violation_trusts_classifier_even_if_llm_disagrees():
    llm = FakeLLM({"compliant": True, "violations": [], "explanation": "LLM thinks it is fine", "severity": "none"})
    result = audit_chunk_hybrid(
        "f", "code", "summary", POLICIES,
        classifier=FakeClassifier([0.95, 0.1, 0.1]), llm_audit=llm,
    )
    assert result["compliant"] is False          # classifier verdict wins
    assert result["severity"] == "critical"
    assert result["decision_path"] == f"[Hybrid/{ROUTE_VIOLATION}]"
    assert result["classifier_confidence"] == 0.95
    assert len(llm.calls) == 1                   # LLM still writes the explanation
    assert "CodeBERT" in llm.calls[0]["hint"]
    # The LLM returned no violations, so the engine names the flagged policy.
    assert result["violations"] == ["Violates policy: policy A"]


# A confidently clean chunk has no violations and severity 'none'.
def test_fast_path_compliant_skips_violations_list():
    llm = FakeLLM()
    result = audit_chunk_hybrid(
        "f", "code", "summary", POLICIES,
        classifier=FakeClassifier([0.01, 0.02, 0.03]), llm_audit=llm,
    )
    assert result["compliant"] is True
    assert result["violations"] == []
    assert result["severity"] == "none"


# In the uncertain band the LLM verdict is used.
def test_ambiguous_case_lets_llm_decide():
    llm = FakeLLM({"compliant": True, "violations": [], "explanation": "fine after review", "severity": "none"})
    result = audit_chunk_hybrid(
        "f", "code", "summary", POLICIES,
        classifier=FakeClassifier([0.5, 0.1, 0.1]), llm_audit=llm,
    )
    assert result["compliant"] is True           # LLM verdict is used in the ambiguous band
    assert result["decision_path"] == f"[Hybrid/{ROUTE_AMBIGUOUS}]"
    assert "decide the verdict yourself" in llm.calls[0]["hint"]


# With explain_fast_path=False a confident chunk makes no LLM call.
def test_explain_fast_path_disabled_skips_llm():
    llm = FakeLLM()
    result = audit_chunk_hybrid(
        "f", "code", "summary", POLICIES,
        classifier=FakeClassifier([0.99, 0.1, 0.1]), llm_audit=llm,
        explain_fast_path=False,
    )
    assert llm.calls == []                       # no API call -> cost saving
    assert result["compliant"] is False
    assert "Violates policy: policy A" in result["violations"][0]


# Without a classifier the LLM decides alone.
def test_llm_only_mode_without_classifier():
    llm = FakeLLM({"compliant": True, "violations": [], "explanation": "ok", "severity": "none"})
    result = audit_chunk_hybrid("f", "code", "summary", POLICIES, classifier=None, llm_audit=llm)
    assert result["decision_path"] == "[LLM-only]"
    assert result["classifier_confidence"] is None
    assert llm.calls[0]["hint"] == ""


# A severity outside the allowed set becomes 'unknown'.
def test_invalid_severity_is_normalised():
    llm = FakeLLM({"compliant": False, "violations": ["x"], "explanation": "e", "severity": "EXTREME"})
    result = audit_chunk_hybrid("f", "code", "summary", POLICIES, classifier=None, llm_audit=llm)
    assert result["severity"] == "unknown"
