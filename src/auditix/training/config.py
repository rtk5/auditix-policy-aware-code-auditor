"""Training hyperparameters and file locations.

The values are the ones from the ``CFG`` dictionary in ``fineTuned_model.ipynb``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from auditix.config import PROJECT_ROOT


@dataclass
class TrainConfig:
    """All knobs for one fine-tuning run. Defaults match the notebook."""

    # ── Model ───────────────────────────────────────────────────────────────
    base_model: str = "microsoft/codebert-base"
    num_labels: int = 2                    # 0 = compliant, 1 = violation
    max_length: int = 512                  # CodeBERT's context limit in tokens

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora_r: int = 16                       # rank of the low-rank update matrices
    lora_alpha: int = 32                   # scaling: update is multiplied by alpha / r
    lora_dropout: float = 0.1
    lora_targets: list[str] = field(default_factory=lambda: ["query", "key"])  # attention layers to adapt

    # ── Training ────────────────────────────────────────────────────────────
    epochs: int = 5
    batch_size: int = 16
    lr: float = 2e-4                       # learning rate (typical for LoRA)
    weight_decay: float = 0.01             # L2-style penalty against overfitting
    warmup_ratio: float = 0.1              # first 10% of steps ramp the learning rate up
    grad_accum: int = 2                    # effective batch = 16 * 2 = 32
    fp16: bool = True                      # half precision on GPU (ignored on CPU)
    seed: int = 42                         # fixes shuffling and splits for repeatability

    # ── Splits ──────────────────────────────────────────────────────────────
    train_ratio: float = 0.80
    val_ratio: float = 0.10                # test = remainder (0.10)

    # ── Paths ───────────────────────────────────────────────────────────────
    output_dir: Path = PROJECT_ROOT / "models" / "clf_checkpoints"   # per-epoch checkpoints
    save_path: Path = PROJECT_ROOT / "models" / "policy_classifier"  # final adapter
    dataset_cache: Path = PROJECT_ROOT / "models" / "ft_dataset.json"
    max_per_source: int = 3000             # cap on samples taken from each dataset

    def to_dict(self) -> dict:
        """Return the config as JSON-safe values (Path objects become strings)."""
        data = asdict(self)
        for key in ("output_dir", "save_path", "dataset_cache"):
            data[key] = str(data[key])
        return data
