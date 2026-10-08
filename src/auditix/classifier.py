"""Inference wrapper for the fine-tuned CodeBERT + LoRA policy classifier.

The classifier scores a *(code, policy)* pair and returns ``p(violation)``.
Input is encoded as ``[CLS] code [SEP] policy [SEP]``, so the model can attend
across both texts. Label 1 means VIOLATION and label 0 means COMPLIANT.

Torch and transformers are imported inside the functions, so the rest of the
package (chunking, retrieval, LLM auditing, reports) works without them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from auditix.config import CLF_MAX_LENGTH

logger = logging.getLogger(__name__)

VIOLATION_LABEL = 1  # index of the VIOLATION logit in the saved model


class PolicyClassifier:
    """A loaded CodeBERT classifier ready to score (code, policy) pairs."""

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device  # "cuda" or "cpu"

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | None = None) -> "PolicyClassifier":
        """Load the tokenizer and model from ``path`` and move the model to the device."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # GPU if possible
        path = str(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        # The folder holds a LoRA adapter; transformers loads the base CodeBERT
        # weights named in adapter_config.json and attaches the adapter.
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device).eval()  # eval(): turns off dropout for inference
        logger.info("Loaded CodeBERT policy classifier from %s on %s", path, device)
        return cls(model, tokenizer, device)

    def violation_probability(self, code: str, policy: str) -> float:
        """Return p(VIOLATION) for one (code, policy) pair, in [0, 1]."""
        import torch

        # Encode code and policy as one sentence pair, padded/truncated to a fixed length.
        enc = self.tokenizer(
            code,
            policy,
            max_length=CLF_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():  # no gradients needed for inference: saves memory and time
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0]  # logits -> probabilities
        return float(probs[VIOLATION_LABEL].item())


def load_classifier(path: str | Path, enabled: bool = True) -> PolicyClassifier | None:
    """Load the classifier, or return ``None`` so the pipeline falls back to LLM-only.

    This mirrors the notebook's graceful degradation: a missing folder, a missing
    torch install, or a corrupt checkpoint all lead to a warning, not a crash.
    """
    if not enabled:
        logger.info("Classifier disabled by configuration — running LLM-only")
        return None
    path = Path(path)
    if not path.is_dir():
        logger.warning("Classifier folder not found: %s — running LLM-only", path)
        return None
    try:
        return PolicyClassifier.from_pretrained(path)
    except Exception as exc:  # broad on purpose: any failure should degrade, not crash
        logger.warning("Could not load classifier (%s) — running LLM-only", exc)
        return None
