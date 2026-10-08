"""Command-line interface.

    auditix audit-local  path/to/project  -o audit_output
    auditix audit-repo   https://github.com/org/repo -o audit_output
    auditix build-index
    auditix train        --help          (needs requirements-train.txt)

Run ``auditix --help`` or ``python -m auditix --help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from auditix import __version__
from auditix.config import Settings, load_dotenv_file


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", type=Path, help="folder for audit_results.json / audit_report.pdf")
    parser.add_argument("--policies", type=Path, help="policy file (.txt, .md or .pdf)")
    parser.add_argument("--classifier", type=Path, help="folder with the fine-tuned CodeBERT adapter")
    parser.add_argument("--no-classifier", action="store_true", help="LLM-only mode (no CodeBERT)")
    parser.add_argument(
        "--no-explain-fast-path",
        action="store_true",
        help="skip the LLM for high-confidence chunks (saves API calls, template explanation)",
    )
    parser.add_argument("--pause", type=float, default=1.0, help="seconds to wait between LLM calls")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    overrides: dict = {}
    if getattr(args, "output", None):
        overrides["output_dir"] = args.output
    if getattr(args, "policies", None):
        overrides["policies_path"] = args.policies
    if getattr(args, "classifier", None):
        overrides["classifier_path"] = args.classifier
    if getattr(args, "no_classifier", False):
        overrides["use_classifier"] = False
    if getattr(args, "no_explain_fast_path", False):
        overrides["explain_fast_path"] = False
    return replace(settings, **overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auditix", description="Policy-aware code compliance auditor")
    parser.add_argument("--version", action="version", version=f"auditix {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_local = sub.add_parser("audit-local", help="audit a folder of Python files")
    p_local.add_argument("path", type=Path, help="folder to audit")
    _add_common_flags(p_local)

    p_repo = sub.add_parser("audit-repo", help="clone a git repository and audit it")
    p_repo.add_argument("url", help="git URL (https)")
    _add_common_flags(p_repo)

    p_index = sub.add_parser("build-index", help="embed the policies and save the FAISS index")
    p_index.add_argument("--policies", type=Path, help="policy file to index")
    p_index.add_argument("--index-dir", type=Path, help="where to save index.faiss")

    p_train = sub.add_parser("train", help="fine-tune the CodeBERT classifier (GPU recommended)")
    p_train.add_argument("--output-dir", type=Path, help="where to save the adapter")
    p_train.add_argument("--cache", type=Path, help="dataset cache file (JSON)")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--max-per-source", type=int, default=None, help="cap samples per dataset")
    p_train.add_argument("--rebuild-dataset", action="store_true", help="ignore the cache and reload datasets")
    return parser


def _run_audit(args: argparse.Namespace, *, repo: bool) -> int:
    from auditix.pipeline import build_context, run_audit_local, run_audit_repo

    settings = _settings_from_args(args)
    ctx = build_context(settings, pause_seconds=args.pause)
    if repo:
        results = run_audit_repo(args.url, ctx=ctx)
    else:
        results = run_audit_local(args.path, ctx=ctx)

    s = results["summary"]
    print("\n" + "=" * 60)
    print("FULL AUDIT COMPLETE")
    print("=" * 60)
    print(f"  Mode             : {results['audit_mode']}")
    print(f"  Files audited    : {s['files']}")
    print(f"  Chunks analyzed  : {s['chunks']}")
    print(f"  Total violations : {s['violations']}")
    print(f"  Critical issues  : {s['critical']}")
    print(f"  Compliance rate  : {s['compliance_rate']}%")
    print(f"  Output folder    : {settings.output_dir}")
    print("=" * 60)
    return 0


def _run_build_index(args: argparse.Namespace) -> int:
    from auditix.embeddings import GeminiEmbedder
    from auditix.policies import load_policies
    from auditix.policy_index import PolicyIndex

    settings = Settings.from_env()
    policies_path = args.policies or settings.policies_path
    index_dir = args.index_dir or settings.index_dir
    policies = load_policies(policies_path)
    index = PolicyIndex.load_or_build(policies, index_dir, GeminiEmbedder())
    print(f"Indexed {index.size} policies into {index_dir}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    try:
        from auditix.training.train import main as train_main
    except ImportError as exc:  # torch / transformers not installed
        print(f"Training dependencies are missing: {exc}", file=sys.stderr)
        print("Install them with: pip install -r requirements-train.txt", file=sys.stderr)
        return 1
    return train_main(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv_file()  # reads ./.env from the project root if present
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "audit-local":
            return _run_audit(args, repo=False)
        if args.command == "audit-repo":
            return _run_audit(args, repo=True)
        if args.command == "build-index":
            return _run_build_index(args)
        if args.command == "train":
            return _run_train(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
