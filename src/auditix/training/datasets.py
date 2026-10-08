"""Build the (code, policy, label) training set from public datasets.

Everything in this file is pure Python except the three loaders, which import
``datasets`` / ``huggingface_hub`` only when they are called. That keeps the
logic easy to unit-test.

Labels: 1 = VIOLATION, 0 = COMPLIANT.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from auditix.training.config import TrainConfig

logger = logging.getLogger(__name__)

# Keyword -> policy rules used to pair an unlabelled snippet with a policy.
# The first rule whose keyword appears in the code wins (same order as notebook).
KEYWORD_RULES: list[tuple[list[str], str]] = [
    (["payment", "charge", "transaction", "invoice", "billing"],
     "All payment transactions must be logged before and after execution using AuditLogger."),
    (["discount", "offer", "promo", "coupon", "voucher"],
     "Discounts must not be hardcoded and must be configurable via the offer system."),
    (["email", "smtp", "send_mail", "sendgrid", "mailchimp"],
     "All user communications must go through a centralized communication service."),
    (["customer", "user", "pii", "personal", "gdpr", "email"],
     "Customer PII must only be accessed via Customer service classes."),
    (["order", "status", "state", "transition", "workflow"],
     "All order state transitions must follow the defined workflow system."),
    (["log", "logger", "audit", "record", "trace"],
     "Critical operations must include structured logging for auditing purposes."),
    (["db", "database", "objects.filter", "objects.update", "queryset", "orm"],
     "No direct database updates to payment or order status are allowed."),
    (["stripe", "paypal", "braintree", "square", "adyen"],
     "Only approved payment providers may be used via PaymentService."),
]


def stable_index(text: str, modulo: int) -> int:
    """Deterministic hash. Python's built-in ``hash()`` changes between runs, so we avoid it."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % modulo


def assign_policy(code_text: str, policies: Sequence[str]) -> str:
    """Pick the most plausible policy for a snippet using keyword overlap (no API calls)."""
    lowered = code_text.lower()
    for keywords, policy in KEYWORD_RULES:
        if policy in policies and any(kw in lowered for kw in keywords):
            return policy
    return policies[stable_index(code_text, len(policies))]


def clean_code(text: Any, max_chars: int = 1500) -> str:
    if not text or not isinstance(text, str):
        return ""
    return text.strip()[:max_chars]


def make_sample(code: str, policy: str, label: int, source: str) -> dict[str, Any]:
    return {"code": code, "policy": policy, "label": int(label), "source": source}


# ── Loaders (network: Hugging Face) ──────────────────────────────────────────

def load_securecode_v2(policies: Sequence[str], max_samples: int = 1000, token: str | None = None) -> list[dict]:
    """SecureCode v2: assistant turns in JSONL files. ``category == "secure"`` -> label 0."""
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id="scthornton/securecode-v2", repo_type="dataset", token=token)
    logger.info("SecureCode v2 snapshot: %s", repo_path)

    samples: list[dict] = []
    for root, _, files in os.walk(repo_path):
        for name in files:
            if not name.endswith(".jsonl"):
                continue
            with open(os.path.join(root, name), encoding="utf-8") as fh:
                for line in fh:
                    if len(samples) >= max_samples:
                        return samples
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    code = "".join(
                        m.get("value", "") + "\n"
                        for m in data.get("conversations", []) or []
                        if isinstance(m, dict) and m.get("from") == "assistant"
                    ).strip()
                    if len(code) < 30:
                        continue
                    label = 0 if (data.get("metadata") or {}).get("category") == "secure" else 1
                    samples.append(make_sample(code, assign_policy(code, policies), label, "securecode-v2"))
    return samples


def load_codexglue_defect(policies: Sequence[str], max_samples: int = 3000) -> list[dict]:
    """CodeXGLUE defect detection (Devign). ``target`` 1 = defective -> violation."""
    from datasets import load_dataset

    ds = load_dataset("code_x_glue_cc_defect_detection", split="train")
    samples: list[dict] = []
    for row in ds.select(range(min(max_samples, len(ds)))):
        code = clean_code(row.get("func", ""))
        if len(code) < 30:
            continue
        samples.append(make_sample(code, assign_policy(code, policies), min(int(row.get("target", 0)), 1), "codexglue-defect"))
    logger.info("CodeXGLUE defect: %d samples", len(samples))
    return samples


def load_clean_python(policies: Sequence[str], max_samples: int = 3000) -> list[dict]:
    """Real-world Python files used as the COMPLIANT baseline (label 0).

    Streams ``codeparrot/github-code`` and keeps files of at least 100 characters
    that contain a ``def``. This is a weak assumption (see explanation.md).
    """
    from datasets import load_dataset

    stream = load_dataset("codeparrot/github-code", split="train", streaming=True)
    samples: list[dict] = []
    for row in stream.take(max_samples * 3):
        if len(samples) >= max_samples:
            break
        code = clean_code(row.get("code") or row.get("content") or "")
        if len(code) < 100 or "def " not in code:
            continue
        samples.append(make_sample(code, assign_policy(code, policies), 0, "clean-python"))
    logger.info("Clean Python: %d samples", len(samples))
    return samples


# ── Pure dataset preparation ─────────────────────────────────────────────────

def merge_dedupe_balance(samples: Sequence[dict], seed: int = 42) -> list[dict]:
    """Remove empty rows, drop duplicate code across ALL sources, and undersample to 50/50.

    The notebook deduplicated inside one loader only, so the same snippet could
    appear in several sources and leak between train and test. Deduping here, on
    the full merged list, fixes that.
    """
    rng = random.Random(seed)
    usable = [s for s in samples if s.get("code", "").strip() and s.get("policy", "").strip()]

    seen: set[str] = set()
    unique: list[dict] = []
    for s in usable:
        if s["code"] not in seen:
            seen.add(s["code"])
            unique.append(s)
    logger.info("Samples: %d raw, %d after cross-source dedupe", len(usable), len(unique))

    violations = [s for s in unique if s["label"] == 1]
    compliant = [s for s in unique if s["label"] == 0]
    n = min(len(violations), len(compliant))
    if n == 0:
        raise ValueError("One class is empty; check that at least one dataset loaded.")
    rng.shuffle(violations)
    rng.shuffle(compliant)
    balanced = violations[:n] + compliant[:n]
    rng.shuffle(balanced)
    return balanced


def split_dataset(balanced: Sequence[dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, list[dict]]:
    n = len(balanced)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": list(balanced[:n_train]),
        "val": list(balanced[n_train:n_train + n_val]),
        "test": list(balanced[n_train + n_val:]),
    }


def build_dataset(
    policies: Sequence[str],
    cfg: TrainConfig,
    *,
    token: str | None = None,
    rebuild: bool = False,
) -> dict[str, list[dict]]:
    """Load every source, merge, balance and split. Results are cached as JSON."""
    cache = Path(cfg.dataset_cache)
    if cache.exists() and not rebuild:
        logger.info("Using cached dataset %s", cache)
        return json.loads(cache.read_text(encoding="utf-8"))

    random.seed(cfg.seed)
    cap = cfg.max_per_source
    samples = (
        load_securecode_v2(policies, max_samples=min(cap, 1000), token=token)
        + load_codexglue_defect(policies, max_samples=cap)
        + load_clean_python(policies, max_samples=cap)
    )
    balanced = merge_dedupe_balance(samples, seed=cfg.seed)
    splits = split_dataset(balanced, cfg.train_ratio, cfg.val_ratio)

    summary = Counter(s["source"] for s in samples)
    logger.info("Source counts: %s", dict(summary))
    logger.info("Splits — train %d, val %d, test %d", len(splits["train"]), len(splits["val"]), len(splits["test"]))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(splits), encoding="utf-8")
    return splits
