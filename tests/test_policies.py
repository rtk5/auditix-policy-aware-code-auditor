from pathlib import Path

import pytest

from auditix.policies import load_policies, parse_policy_text

REPO_ROOT = Path(__file__).resolve().parents[1]


# Numbering, bullets and '#' headings are removed; short lines are ignored.
def test_parse_policy_text_strips_numbering_bullets_and_comments():
    text = """
    # Heading that should be ignored
    1. All payment transactions must be logged using AuditLogger.
    - Discounts must not be hardcoded anywhere in the codebase.
    * Customer PII must only be accessed via Customer service classes.
    short
    """
    assert parse_policy_text(text) == [
        "All payment transactions must be logged using AuditLogger.",
        "Discounts must not be hardcoded anywhere in the codebase.",
        "Customer PII must only be accessed via Customer service classes.",
    ]


# Repeats that differ only in case are kept once.
def test_parse_policy_text_removes_case_insensitive_duplicates():
    text = "Logs must be structured for auditing.\nlogs must be STRUCTURED for auditing."
    assert len(parse_policy_text(text)) == 1


# The bundled policies file has the 12 expected statements.
def test_shipped_policies_file_has_twelve_policies():
    policies = load_policies(REPO_ROOT / "policies" / "policies.txt")
    assert len(policies) == 12
    assert all(len(p) > 20 for p in policies)


# A missing file raises FileNotFoundError.
def test_missing_policy_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policies(tmp_path / "nope.txt")


# A file with only comments yields no policies, which is an error.
def test_empty_policy_file_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_policies(f)
