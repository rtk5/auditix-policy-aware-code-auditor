"""Central configuration for the auditor.

Every tunable value lives here so that the rest of the code never hard-codes
model names, thresholds or paths. Values can be overridden with environment
variables (see ``.env.example``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = the folder that contains ``src/``, ``policies/`` and ``models/``.
# This file lives in src/auditix/config.py, so parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_groq_api_key() -> str | None:
    """Groq key. ``GROK_API_KEY`` is accepted because the notebooks used that name."""
    return os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


# ── LLM endpoints and models ────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_SUMMARY_MODEL = "llama-3.1-8b-instant"       # cheap summaries for retrieval
GROQ_AUDIT_MODEL = "moonshotai/kimi-k2-instruct"  # deep compliance reasoning
GROQ_TIMEOUT_SECONDS = 60

GEMINI_EMBED_MODEL = "models/gemini-embedding-2-preview"  # 3072-dim vectors

# ── Hybrid decision thresholds (see hybrid_engine.py) ───────────────────────
AMBIGUITY_LOW = 0.35   # p(violation) below this -> fast-path COMPLIANT
AMBIGUITY_HIGH = 0.65  # p(violation) above this -> fast-path VIOLATION
CRITICAL_THRESHOLD = 0.92  # severity bands used for classifier-only verdicts

# ── Retrieval / classifier ──────────────────────────────────────────────────
TOP_K_POLICIES = 3       # FAISS neighbours passed to the classifier
CLF_MAX_LENGTH = 512     # CodeBERT context window in tokens

# ── Files that are skipped when scanning a repository ───────────────────────
SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", "venv", "node_modules", "migrations"})
MIN_FILE_CHARS = 50      # files shorter than this are not audited


@dataclass
class Settings:
    """Runtime settings. Build one with :func:`Settings.from_env`."""

    policies_path: Path = PROJECT_ROOT / "policies" / "policies.txt"
    classifier_path: Path = PROJECT_ROOT / "models" / "policy_classifier"
    index_dir: Path = PROJECT_ROOT / "models" / "policy_index"
    output_dir: Path = PROJECT_ROOT / "audit_output"
    top_k: int = TOP_K_POLICIES
    # When False, fast-path chunks skip the LLM and get a template explanation.
    # When True (the notebook behaviour) the LLM always writes the explanation.
    explain_fast_path: bool = True
    use_classifier: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        base = cls()
        return cls(
            policies_path=Path(os.getenv("AUDITIX_POLICIES_PATH", base.policies_path)),
            classifier_path=Path(os.getenv("AUDITIX_CLASSIFIER_PATH", base.classifier_path)),
            index_dir=Path(os.getenv("AUDITIX_INDEX_DIR", base.index_dir)),
            output_dir=Path(os.getenv("AUDITIX_OUTPUT_DIR", base.output_dir)),
            top_k=base.top_k,
            explain_fast_path=_env_bool("AUDITIX_EXPLAIN_FAST_PATH", True),
            use_classifier=_env_bool("AUDITIX_USE_CLASSIFIER", True),
        )


def load_dotenv_file(path: str | Path | None = None) -> bool:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Existing environment variables always win, so a key exported in the shell
    is never overwritten by the file. Returns True if a file was read.
    Only the simple ``KEY=VALUE`` form is supported (comments with ``#`` and
    optional quotes are handled), which is all ``.env.example`` uses.
    """
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return False
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    return True
