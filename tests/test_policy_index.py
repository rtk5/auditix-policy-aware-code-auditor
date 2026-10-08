import numpy as np

from auditix.policy_index import PolicyIndex

POLICIES = ["payments must be logged", "discounts must be configurable", "emails need opt-out"]
VECTORS = {
    POLICIES[0]: [1.0, 0.0, 0.0],
    POLICIES[1]: [0.0, 1.0, 0.0],
    POLICIES[2]: [0.0, 0.0, 1.0],
}


class FakeEmbedder:
    """Deterministic stand-in for Gemini: each policy has a known one-hot vector."""

    def __init__(self):
        self.doc_calls = 0

    def embed_documents(self, texts):
        self.doc_calls += 1
        return np.asarray([VECTORS[t] for t in texts], dtype="float32")

    def embed_query(self, text):
        return np.asarray([VECTORS[text]], dtype="float32")


def test_search_returns_closest_policy_first(tmp_path):
    emb = FakeEmbedder()
    index = PolicyIndex.build(POLICIES, emb)
    assert index.size == 3
    assert index.search(POLICIES[2], top_k=1) == [POLICIES[2]]
    assert index.search(POLICIES[1], top_k=2)[0] == POLICIES[1]


def test_top_k_is_capped_at_index_size():
    index = PolicyIndex.build(POLICIES, FakeEmbedder())
    assert len(index.search(POLICIES[0], top_k=10)) == 3


def test_save_then_load_or_build_reuses_cache(tmp_path):
    first = FakeEmbedder()
    PolicyIndex.load_or_build(POLICIES, tmp_path, first)
    assert first.doc_calls == 1

    second = FakeEmbedder()
    reloaded = PolicyIndex.load_or_build(POLICIES, tmp_path, second)
    assert second.doc_calls == 0                 # no re-embedding when policies unchanged
    assert reloaded.search(POLICIES[0], 1) == [POLICIES[0]]


def test_changed_policies_trigger_rebuild(tmp_path):
    PolicyIndex.load_or_build(POLICIES, tmp_path, FakeEmbedder())
    emb = FakeEmbedder()
    PolicyIndex.load_or_build(POLICIES[:2], tmp_path, emb)
    assert emb.doc_calls == 1
