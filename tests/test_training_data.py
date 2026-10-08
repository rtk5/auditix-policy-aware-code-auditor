from auditix.policies import load_policies
from auditix.training.datasets import (
    assign_policy,
    merge_dedupe_balance,
    make_sample,
    split_dataset,
    stable_index,
)
from pathlib import Path

from auditix.training.config import TrainConfig

POLICIES = load_policies(Path(__file__).resolve().parents[1] / "policies" / "policies.txt")


def test_assign_policy_uses_keywords_first():
    code = "def charge_card(amount): create_payment_transaction(amount)"
    assert assign_policy(code, POLICIES).startswith("All payment transactions")


def test_assign_policy_is_deterministic_for_unmatched_code():
    code = "def zzz(): return 42"
    assert assign_policy(code, POLICIES) == assign_policy(code, POLICIES)
    assert stable_index(code, 12) == stable_index(code, 12)


def test_merge_removes_duplicates_across_sources_and_balances():
    samples = [
        make_sample("code A", POLICIES[0], 1, "src1"),
        make_sample("code A", POLICIES[0], 1, "src2"),  # same code from another source
        make_sample("code B", POLICIES[0], 0, "src1"),
        make_sample("code C", POLICIES[1], 0, "src1"),
        make_sample("code D", POLICIES[1], 1, "src2"),
        make_sample("   ", POLICIES[1], 1, "src2"),      # empty -> dropped
    ]
    balanced = merge_dedupe_balance(samples, seed=1)
    codes = [s["code"] for s in balanced]
    assert len(codes) == len(set(codes)) == 4
    assert sum(s["label"] for s in balanced) == 2        # 50 / 50


def test_merge_raises_when_one_class_is_empty():
    import pytest

    with pytest.raises(ValueError):
        merge_dedupe_balance([make_sample("only violations", POLICIES[0], 1, "x")])


def test_split_ratios_cover_all_samples():
    data = [make_sample(f"c{i}", POLICIES[0], i % 2, "x") for i in range(100)]
    splits = split_dataset(data, 0.8, 0.1)
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (80, 10, 10)


def test_default_train_config_matches_notebook():
    cfg = TrainConfig()
    assert (cfg.lora_r, cfg.lora_alpha, cfg.lr, cfg.epochs, cfg.grad_accum) == (16, 32, 2e-4, 5, 2)
    assert cfg.lora_targets == ["query", "key"]
