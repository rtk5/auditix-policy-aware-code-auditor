"""End-to-end orchestration: files -> chunks -> summary -> policies -> verdict -> report.

This module is the Python equivalent of ``run_compliance_audit()`` and the repo
loop in ``complete_code.ipynb``. Each stage is a separate function so it can be
tested or replaced on its own.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auditix import reporting
from auditix.chunker import parse_code_chunks
from auditix.classifier import PolicyClassifier, load_classifier
from auditix.config import MIN_FILE_CHARS, SKIP_DIRS, Settings
from auditix.embeddings import GeminiEmbedder
from auditix.hybrid_engine import audit_chunk_hybrid
from auditix.llm.groq_client import audit_chunk, summarize_chunk
from auditix.policies import load_policies
from auditix.policy_index import PolicyIndex

logger = logging.getLogger(__name__)


@dataclass
class AuditContext:
    """Everything the per-chunk audit needs. Built once per run."""

    settings: Settings
    policies: list[str]
    index: PolicyIndex
    classifier: PolicyClassifier | None
    summarize: Callable[[str], str] = summarize_chunk
    llm_audit: Callable[..., dict[str, Any]] = audit_chunk
    pause_seconds: float = 1.0  # gentle pacing between LLM calls (rate limits)

    @property
    def audit_mode(self) -> str:
        return "hybrid" if self.classifier is not None else "llm-only"


def build_context(
    settings: Settings | None = None,
    *,
    embedder=None,
    classifier: PolicyClassifier | None | str = "auto",
    pause_seconds: float = 1.0,
) -> AuditContext:
    """Load policies, build or reuse the FAISS index, and load the classifier.

    ``classifier="auto"`` loads it from ``settings.classifier_path`` when present.
    Pass ``None`` to force LLM-only mode, or a ready object to inject one.
    """
    settings = settings or Settings.from_env()
    policies = load_policies(settings.policies_path)
    embedder = embedder or GeminiEmbedder()
    index = PolicyIndex.load_or_build(policies, settings.index_dir, embedder)
    if classifier == "auto":
        classifier = load_classifier(settings.classifier_path, enabled=settings.use_classifier)
    logger.info("Audit mode: %s (%d policies indexed)", "hybrid" if classifier else "llm-only", len(policies))
    return AuditContext(
        settings=settings,
        policies=policies,
        index=index,
        classifier=classifier,
        pause_seconds=pause_seconds,
    )


def audit_source(source_code: str, filename: str, ctx: AuditContext) -> dict[str, Any]:
    """Audit one Python file and return its per-file report."""
    logger.info("Auditing %s", filename)
    chunks = parse_code_chunks(source_code, filename)
    logger.info("  found %d chunks", len(chunks))

    report: dict[str, Any] = {
        "file": filename,
        "chunks": [],
        "summary": {"total": 0, "violations": 0, "critical": 0},
    }
    for chunk in chunks:
        summary = ctx.summarize(chunk.code)                              # step 3: LLaMA summary
        relevant = ctx.index.search(summary, ctx.settings.top_k)         # step 4: FAISS top-k
        verdict = audit_chunk_hybrid(                                    # steps 5–7: CodeBERT + Kimi
            chunk.name,
            chunk.code,
            summary,
            relevant,
            classifier=ctx.classifier,
            llm_audit=ctx.llm_audit,
            explain_fast_path=ctx.settings.explain_fast_path,
        )

        report["chunks"].append({
            "name": chunk.name,
            "type": chunk.kind,
            "summary": summary,
            "policies_checked": relevant,
            "compliant": verdict["compliant"],
            "violations": verdict["violations"],
            "explanation": verdict["explanation"],
            "severity": verdict["severity"],
            "classifier_confidence": verdict["classifier_confidence"],
            "decision_path": verdict["decision_path"],
        })
        report["summary"]["total"] += 1
        if not verdict["compliant"]:
            report["summary"]["violations"] += 1
        if verdict["severity"] == "critical":
            report["summary"]["critical"] += 1

        if ctx.pause_seconds:
            time.sleep(ctx.pause_seconds)
    return report


def collect_python_files(root: str | Path) -> list[Path]:
    """All ``.py`` files under ``root``, skipping vendor/cache folders."""
    root = Path(root)
    files: list[Path] = []
    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).parts[:-1]
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        files.append(path)
    return sorted(files)


def audit_directory(root: str | Path, ctx: AuditContext) -> dict[str, Any]:
    """Audit every eligible ``.py`` file under ``root`` and aggregate the totals."""
    root = Path(root)
    reports: list[dict[str, Any]] = []
    for path in collect_python_files(root):
        code = path.read_text(encoding="utf-8", errors="ignore")
        if len(code.strip()) < MIN_FILE_CHARS:
            continue
        reports.append(audit_source(code, path.relative_to(root).as_posix(), ctx))
    return _aggregate(reports, ctx)


def _aggregate(reports: Sequence[dict[str, Any]], ctx: AuditContext) -> dict[str, Any]:
    total = sum(r["summary"]["total"] for r in reports)
    violations = sum(r["summary"]["violations"] for r in reports)
    critical = sum(r["summary"]["critical"] for r in reports)
    rate = round((total - violations) / total * 100) if total else 0
    return {
        "audit_mode": ctx.audit_mode,
        "reports": list(reports),
        "summary": {
            "files": len(reports),
            "chunks": total,
            "violations": violations,
            "critical": critical,
            "compliance_rate": rate,
        },
    }


def write_outputs(results: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write ``audit_results.json`` and ``audit_report.pdf`` into ``output_dir``."""
    output_dir = Path(output_dir)
    return {
        "json": reporting.write_json(results, output_dir / "audit_results.json"),
        "pdf": reporting.write_pdf(results, output_dir / "audit_report.pdf"),
    }


def run_audit_local(
    code_dir: str | Path,
    output_dir: str | Path | None = None,
    ctx: AuditContext | None = None,
    **ctx_kwargs: Any,
) -> dict[str, Any]:
    """Audit a folder on disk and write the reports."""
    ctx = ctx or build_context(**ctx_kwargs)
    results = audit_directory(code_dir, ctx)
    results["source"] = str(code_dir)
    out = Path(output_dir) if output_dir else ctx.settings.output_dir
    write_outputs(results, out)
    logger.info("Reports written to %s", out)
    return results


def clone_repo(repo_url: str, destination: str | Path) -> Path:
    """Shallow-clone ``repo_url`` into ``destination`` using the git CLI."""
    destination = Path(destination)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(destination)], check=True)
    return destination


def run_audit_repo(
    repo_url: str,
    output_dir: str | Path | None = None,
    ctx: AuditContext | None = None,
    **ctx_kwargs: Any,
) -> dict[str, Any]:
    """Clone a GitHub repository to a temporary folder, audit it, write reports."""
    ctx = ctx or build_context(**ctx_kwargs)
    with tempfile.TemporaryDirectory(prefix="auditix-") as tmp:
        clone_dir = clone_repo(repo_url, Path(tmp) / "repo")
        results = audit_directory(clone_dir, ctx)
    results["source"] = repo_url
    out = Path(output_dir) if output_dir else ctx.settings.output_dir
    write_outputs(results, out)
    logger.info("Reports written to %s", out)
    return results


__all__ = [
    "AuditContext",
    "audit_directory",
    "audit_source",
    "build_context",
    "collect_python_files",
    "run_audit_local",
    "run_audit_repo",
    "write_outputs",
]
