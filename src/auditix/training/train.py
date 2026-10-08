"""Fine-tune CodeBERT with LoRA on (code, policy) pairs.

Requires the packages in ``requirements-train.txt``. Run it with::

    auditix train --epochs 5
    # or
    python -m auditix.training.train --help

The output folder holds a LoRA adapter (``adapter_model.safetensors``), the
tokenizer, ``training_config.json`` and ``metrics.json``. ``PolicyClassifier``
in ``auditix.classifier`` loads it again for inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from auditix.config import Settings
from auditix.policies import load_policies
from auditix.training.config import TrainConfig
from auditix.training.datasets import build_dataset

logger = logging.getLogger(__name__)
LABEL_NAMES = ["COMPLIANT", "VIOLATION"]  # index 0 and 1, matching the model's labels


class PolicyPairDataset(Dataset):
    """Encodes each sample as ``[CLS] code [SEP] policy [SEP]``."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenise one sample. The tokenizer adds the special tokens between code and policy."""
        s = self.samples[idx]
        enc = self.tokenizer(
            s["code"],
            s["policy"],
            max_length=self.max_length,
            padding="max_length",   # every sample has the same length, so batches stack cleanly
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],          # [0]: remove the batch dimension of 1
            "attention_mask": enc["attention_mask"][0],  # 1 for real tokens, 0 for padding
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }


def build_lora_model(cfg: TrainConfig):
    """Load CodeBERT with a 2-way head and wrap it with LoRA. Only the adapter is trainable."""
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=cfg.num_labels,
        id2label={0: "COMPLIANT", 1: "VIOLATION"},
        label2id={"COMPLIANT": 0, "VIOLATION": 1},
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,          # sequence classification: PEFT keeps the head trainable
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets,     # only the query and key projections get adapters
        bias="none",
        inference_mode=False,                # we are training, not just running
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()       # prints the trainable share (about 0.94%)
    return model


def compute_metrics(eval_pred) -> dict[str, float]:
    """Called by the Trainer after each epoch. Returns accuracy and macro-F1 (equal weight per class)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)  # highest logit = predicted class
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
    }


def train_model(cfg: TrainConfig, splits: dict[str, list[dict]]):
    """Train with the notebook's settings. Returns (trainer, tokenizer)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No CUDA GPU detected — training will be very slow")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    train_ds = PolicyPairDataset(splits["train"], tokenizer, cfg.max_length)
    val_ds = PolicyPairDataset(splits["val"], tokenizer, cfg.max_length)
    model = build_lora_model(cfg)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,   # evaluation needs no gradients, so it can be larger
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.grad_accum,      # simulates a batch of 32 on a small GPU
        fp16=cfg.fp16 and device == "cuda",
        evaluation_strategy="epoch",                     # evaluate on validation after every epoch
        save_strategy="epoch",                           # and save a checkpoint each epoch
        load_best_model_at_end=True,                     # finish with the best epoch, not the last
        metric_for_best_model="f1",                      # "best" means highest validation F1
        greater_is_better=True,
        lr_scheduler_type="cosine",                      # learning rate decays smoothly
        logging_steps=10,
        save_total_limit=2,                              # keep only the two newest checkpoints
        report_to="none",                                # no W&B / TensorBoard
        seed=cfg.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    logger.info("Starting fine-tuning on %s", device)
    trainer.train()
    model.to(device)
    return trainer, tokenizer


@torch.no_grad()  # no gradients: this is only measurement
def evaluate_test(model, tokenizer, test_samples: list[dict], cfg: TrainConfig) -> dict[str, Any]:
    """Test-set metrics, computed once after training."""
    device = next(model.parameters()).device  # use whatever device the model is on
    model.eval()  # turn off dropout
    ds = PolicyPairDataset(test_samples, tokenizer, cfg.max_length)
    loader = torch.utils.data.DataLoader(ds, batch_size=32)
    preds, labels = [], []
    for batch in loader:
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        labels.extend(batch["labels"].numpy())

    cm = confusion_matrix(labels, preds, labels=[0, 1])  # rows = true class, columns = predicted
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "confusion_matrix": {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])},
        "report": classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True),
        "n": len(labels),
    }


def save_artifacts(model, tokenizer, cfg: TrainConfig, metrics: dict[str, Any], trainer=None) -> Path:
    """Write the adapter, tokenizer, config, metrics and (if possible) the training curves."""
    out = Path(cfg.save_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)          # saves only the LoRA adapter and classifier head (a few MB)
    tokenizer.save_pretrained(out)
    (out / "training_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if trainer is not None:
        try:
            plot_training_curves(trainer.state.log_history, out / "training_curves.png")
        except ImportError:  # matplotlib is optional
            logger.info("matplotlib not installed — skipping training curves")
    logger.info("Saved classifier to %s", out)
    return out


def plot_training_curves(log_history: list[dict], path: Path) -> None:
    """Draw loss, validation accuracy and validation F1 over training steps."""
    import matplotlib

    matplotlib.use("Agg")  # draw to a file without needing a display
    import matplotlib.pyplot as plt

    train_logs = [log for log in log_history if "loss" in log and "eval_loss" not in log]
    eval_logs = [log for log in log_history if "eval_loss" in log]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot([l["step"] for l in train_logs], [l["loss"] for l in train_logs], label="Train")
    if eval_logs:
        axes[0].plot([l["step"] for l in eval_logs], [l["eval_loss"] for l in eval_logs], marker="o", label="Val")
        axes[1].plot([l["step"] for l in eval_logs], [l.get("eval_accuracy", 0) for l in eval_logs], marker="o")
        axes[2].plot([l["step"] for l in eval_logs], [l.get("eval_f1", 0) for l in eval_logs], marker="o")
    for ax, title in zip(axes, ["Loss", "Val accuracy", "Val F1 (macro)"]):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)  # free the figure memory


def run(cfg: TrainConfig, policies_path: str | Path, *, rebuild_dataset: bool = False, token: str | None = None) -> dict:
    """Full training run: data -> train -> test metrics -> save."""
    torch.manual_seed(cfg.seed)  # repeatable weight initialisation and dropout
    np.random.seed(cfg.seed)
    policies = load_policies(policies_path)
    splits = build_dataset(policies, cfg, token=token or os.getenv("HF_TOKEN"), rebuild=rebuild_dataset)
    trainer, tokenizer = train_model(cfg, splits)
    metrics = evaluate_test(trainer.model, tokenizer, splits["test"], cfg)  # test set is used once, here
    logger.info("Test accuracy %.4f, macro-F1 %.4f", metrics["accuracy"], metrics["macro_f1"])
    save_artifacts(trainer.model, tokenizer, cfg, metrics, trainer)
    return metrics


def main(args: argparse.Namespace | None = None) -> int:
    """Entry point used by ``auditix train``. ``args`` comes from the CLI parser."""
    if args is None:  # run as a script: parse our own arguments
        args = _parser().parse_args()
    cfg = TrainConfig()
    # Only override the defaults for options that were actually given.
    overrides: dict[str, Any] = {}
    if getattr(args, "output_dir", None):
        overrides["save_path"] = Path(args.output_dir)
    if getattr(args, "cache", None):
        overrides["dataset_cache"] = Path(args.cache)
    if getattr(args, "epochs", None):
        overrides["epochs"] = args.epochs
    if getattr(args, "max_per_source", None):
        overrides["max_per_source"] = args.max_per_source
    cfg = replace(cfg, **overrides)

    metrics = run(cfg, Settings.from_env().policies_path, rebuild_dataset=getattr(args, "rebuild_dataset", False))
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                      "confusion_matrix": metrics["confusion_matrix"]}, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    """Argument parser used only when this file is run directly (``python -m ...``)."""
    p = argparse.ArgumentParser(description="Fine-tune the CodeBERT policy classifier")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--epochs", type=int)
    p.add_argument("--max-per-source", type=int)
    p.add_argument("--rebuild-dataset", action="store_true")
    return p


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
