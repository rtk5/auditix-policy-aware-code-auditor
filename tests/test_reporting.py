import json

from auditix.reporting import write_json, write_pdf

RESULTS = {
    "audit_mode": "llm-only",
    "reports": [{
        "file": "payment/utils.py",
        "summary": {"total": 2, "violations": 1, "critical": 0},
        "chunks": [
            {"name": "pay<script>", "summary": "takes payment & logs", "compliant": False,
             "violations": ["Missing AuditLogger <b>"], "explanation": "bad", "severity": "high"},
            {"name": "ok_fn", "summary": "pure function", "compliant": True,
             "violations": [], "explanation": "fine", "severity": "none"},
        ],
    }],
    "summary": {"files": 1, "chunks": 2, "violations": 1, "critical": 0, "compliance_rate": 50},
}


def test_write_json_round_trip(tmp_path):
    path = write_json(RESULTS, tmp_path / "out" / "audit_results.json")
    assert json.loads(path.read_text(encoding="utf-8")) == RESULTS


def test_write_pdf_creates_non_empty_file(tmp_path):
    path = write_pdf(RESULTS, tmp_path / "audit_report.pdf")
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 1000
