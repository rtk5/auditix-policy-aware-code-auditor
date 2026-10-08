"""Fine-tuning pipeline for the CodeBERT + LoRA policy classifier.

Modules:
    config.py    hyperparameters and paths (TrainConfig)
    datasets.py  dataset loaders, policy assignment, dedupe / balance / split (no torch)
    train.py     tokenisation, LoRA model, Trainer, test evaluation (needs torch)
"""
