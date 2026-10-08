"""Write audit results as JSON and as a PDF report (ReportLab)."""

from __future__ import annotations

import json
from html import escape  # ReportLab Paragraphs read HTML-like tags, so user text must be escaped
from pathlib import Path
from typing import Any


def write_json(results: dict[str, Any], path: str | Path) -> Path:
    """Save the full results dict as indented UTF-8 JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # create the output folder if needed
    path.write_text(json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")
    return path


def write_pdf(results: dict[str, Any], path: str | Path) -> Path:
    """Render the same content as the notebook's ``generate_pdf_report()``."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    # Three text styles: red for violations, green for compliant results, plain for details.
    violation_style = ParagraphStyle("ViolationText", parent=styles["Normal"], textColor=colors.red, spaceAfter=4)
    compliant_style = ParagraphStyle("CompliantText", parent=styles["Normal"], textColor=colors.green, spaceAfter=4)
    normal_style = ParagraphStyle("NormalText", parent=styles["Normal"], spaceAfter=4)

    mode_label = "Hybrid (CodeBERT + LLM)" if results.get("audit_mode") == "hybrid" else "LLM-only"
    # ``content`` is the list of flowables ReportLab lays out top to bottom.
    content: list[Any] = [
        Paragraph("Policy-Aware Code Compliance Audit Report", styles["Title"]),
        Paragraph(f"Audit Mode: {escape(mode_label)}", normal_style),
        Spacer(1, 12),
    ]

    for report in results.get("reports", []):  # one section per audited file
        content.append(Paragraph(f"File: {escape(str(report['file']))}", styles["Heading2"]))
        s = report["summary"]
        line = f"Chunks: {s['total']} | Violations: {s['violations']} | Critical: {s['critical']}"
        content.append(Paragraph(escape(line), normal_style))
        content.append(Spacer(1, 6))

        for chunk in report.get("chunks", []):  # one entry per function or class
            name = escape(str(chunk.get("name", "unknown")))
            if chunk.get("compliant", False):
                content.append(Paragraph(f"<b>✔ COMPLIANT — {name}</b>", compliant_style))
            else:
                sev = escape(str(chunk.get("severity", "unknown")).upper())
                content.append(Paragraph(f"<b>✘ VIOLATION [{sev}] — {name}</b>", violation_style))

            # Summary and reason are truncated so one long LLM answer does not fill the page.
            content.append(Paragraph(f"Summary: {escape(str(chunk.get('summary', ''))[:300])}", normal_style))
            content.append(Paragraph(f"Reason: {escape(str(chunk.get('explanation', ''))[:500])}", normal_style))

            violations = chunk.get("violations", [])
            if violations:
                content.append(Paragraph("Policy Violations:", normal_style))
                for v in violations:
                    content.append(Paragraph(f"⚠ {escape(str(v))}", violation_style))
            else:
                content.append(Paragraph("No specific policy violations detected.", compliant_style))
            content.append(Spacer(1, 10))

        content.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))  # divider between files
        content.append(Spacer(1, 12))

    SimpleDocTemplate(str(path)).build(content)  # lays out and writes the PDF
    return path
