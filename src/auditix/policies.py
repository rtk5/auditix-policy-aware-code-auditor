"""Load compliance policies from a text file or a PDF/TXT policy document.

The notebooks hard-coded 12 policies in a Python list. We moved them to
``policies/policies.txt`` so that a non-developer can edit them, and we kept the
dynamic parser from ``fineTuned_model.ipynb`` (Section 4) so any policy document
can be used.
"""

from __future__ import annotations

import re
from pathlib import Path

MIN_POLICY_CHARS = 20


def parse_policy_text(text: str) -> list[str]:
    """Turn raw text into a clean, de-duplicated list of policy statements.

    Rules (same as the notebook parser):
      * skip blank lines and lines shorter than 20 characters;
      * strip numbering (``1.``, ``2)``) and bullet characters (``-``, ``*``, ``•``);
      * skip lines starting with ``#`` (comments / headings);
      * remove duplicates case-insensitively, keeping the first occurrence.
    """
    policies: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) < MIN_POLICY_CHARS or line.startswith("#"):
            continue
        line = re.sub(r"^\d+[.)\s]+", "", line)   # "1. " or "1) "
        line = re.sub(r"^[-•*>]+\s*", "", line)    # bullets
        line = line.strip()
        if len(line) > MIN_POLICY_CHARS:
            policies.append(line)

    seen: set[str] = set()
    unique: list[str] = []
    for policy in policies:
        key = policy.lower()
        if key not in seen:
            seen.add(key)
            unique.append(policy)
    return unique


def load_policies(path: str | Path) -> list[str]:
    """Load policies from ``.txt``, ``.md`` or ``.pdf`` files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # imported lazily: only needed for PDFs
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError("Reading PDF policies requires `pip install pypdf`") from exc
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    policies = parse_policy_text(text)
    if not policies:
        raise ValueError(f"No policy statements found in {path}")
    return policies
