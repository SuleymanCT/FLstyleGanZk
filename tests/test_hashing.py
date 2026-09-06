import torch

from storage.hashing import canonical_hash


def test_key_order_does_not_change_hash():
    state_a = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    state_b = {"b": torch.tensor([3.0]), "a": torch.tensor([1.0, 2.0])}
    assert canonical_hash(state_a) == canonical_hash(state_b)


def test_single_bit_change_changes_hash():
    state_a = {"a": torch.tensor([1.0, 2.0])}
    state_b = {"a": torch.tensor([1.0, 2.0001])}
    assert canonical_hash(state_a) != canonical_hash(state_b)


def test_dtype_cast_to_fp32_does_not_change_hash():
    state_a = {"a": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    state_b = {"a": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    assert canonical_hash(state_a) == canonical_hash(state_b)
