"""attacks/ modülünün SAF (state_dict-only) fonksiyonlarının testleri —
hiçbiri gerçek StyleGAN-XL/pkl/GPU gerektirmez, tamamen sentetik
tensörlerle yerelde GERÇEKTEN test edilir."""

import pytest
import torch

from attacks.conditional_poison import swap_embed_rows
from attacks.random_weights import randomize_state_dict
from attacks.scaled_poison import scale_delta


def _synthetic_state_dict():
    return {
        "mapping.fc0.weight": torch.arange(12.0).reshape(3, 4),
        "mapping.fc0.bias": torch.tensor([1.0, 2.0, 3.0]),
        "synthesis.const": torch.ones(2, 2),
        "num_batches_tracked": torch.tensor(7, dtype=torch.int64),
    }


# --- random_weights.randomize_state_dict ---


def test_randomize_state_dict_preserves_shapes_and_dtypes():
    original = _synthetic_state_dict()
    result = randomize_state_dict(original, generator=torch.Generator().manual_seed(0))
    for key in original:
        assert result[key].shape == original[key].shape
        assert result[key].dtype == original[key].dtype


def test_randomize_state_dict_actually_changes_float_tensors():
    original = _synthetic_state_dict()
    result = randomize_state_dict(original, generator=torch.Generator().manual_seed(0))
    assert not torch.equal(result["mapping.fc0.weight"], original["mapping.fc0.weight"])
    assert not torch.equal(result["mapping.fc0.bias"], original["mapping.fc0.bias"])


def test_randomize_state_dict_leaves_non_float_tensors_untouched():
    original = _synthetic_state_dict()
    result = randomize_state_dict(original, generator=torch.Generator().manual_seed(0))
    assert torch.equal(result["num_batches_tracked"], original["num_batches_tracked"])


def test_randomize_state_dict_deterministic_with_same_seeded_generator():
    original = _synthetic_state_dict()
    r1 = randomize_state_dict(original, generator=torch.Generator().manual_seed(42))
    r2 = randomize_state_dict(original, generator=torch.Generator().manual_seed(42))
    assert torch.equal(r1["mapping.fc0.weight"], r2["mapping.fc0.weight"])


# --- scaled_poison.scale_delta ---


def test_scale_delta_scale_one_equals_site_state():
    site = {"a": torch.tensor([5.0, 5.0]), "b": torch.tensor([1.0])}
    prev = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])}
    result = scale_delta(site, prev, scale=1.0)
    assert torch.allclose(result["a"], site["a"])
    assert torch.allclose(result["b"], site["b"])


def test_scale_delta_scale_zero_equals_prev_global():
    site = {"a": torch.tensor([5.0, 5.0])}
    prev = {"a": torch.tensor([1.0, 2.0])}
    result = scale_delta(site, prev, scale=0.0)
    assert torch.allclose(result["a"], prev["a"])


def test_scale_delta_general_scale_matches_manual_arithmetic():
    site = {"a": torch.tensor([3.0])}
    prev = {"a": torch.tensor([1.0])}
    # delta = 3-1 = 2; poisoned = 1 + 10*2 = 21
    result = scale_delta(site, prev, scale=10.0)
    assert torch.allclose(result["a"], torch.tensor([21.0]))


def test_scale_delta_raises_on_mismatched_keys():
    site = {"a": torch.tensor([1.0])}
    prev = {"b": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="anahtar kümeleri"):
        scale_delta(site, prev, scale=10.0)


# --- conditional_poison.swap_embed_rows ---


def test_swap_embed_rows_swaps_exactly_the_two_rows():
    embed = torch.arange(20.0).reshape(5, 4)  # satır i = [4i, 4i+1, 4i+2, 4i+3]
    state = {"mapping.embed.weight": embed.clone(), "other.key": torch.ones(3)}
    result = swap_embed_rows(state, row_a=0, row_b=4)
    assert torch.equal(result["mapping.embed.weight"][0], embed[4])
    assert torch.equal(result["mapping.embed.weight"][4], embed[0])
    # dokunulmayan satırlar aynen kalmalı
    for i in (1, 2, 3):
        assert torch.equal(result["mapping.embed.weight"][i], embed[i])


def test_swap_embed_rows_leaves_other_keys_untouched():
    embed = torch.arange(20.0).reshape(5, 4)
    other = torch.ones(3)
    state = {"mapping.embed.weight": embed, "other.key": other}
    result = swap_embed_rows(state, row_a=0, row_b=4)
    assert torch.equal(result["other.key"], other)


def test_swap_embed_rows_does_not_mutate_original_dict_entry():
    embed = torch.arange(20.0).reshape(5, 4)
    original = embed.clone()
    state = {"mapping.embed.weight": embed}
    swap_embed_rows(state, row_a=0, row_b=4)
    assert torch.equal(state["mapping.embed.weight"], original)


def test_swap_embed_rows_missing_key_raises_keyerror():
    with pytest.raises(KeyError):
        swap_embed_rows({"other.key": torch.ones(3)}, row_a=0, row_b=1)


def test_swap_embed_rows_same_row_raises_valueerror():
    embed = torch.arange(20.0).reshape(5, 4)
    with pytest.raises(ValueError, match="aynı"):
        swap_embed_rows({"mapping.embed.weight": embed}, row_a=2, row_b=2)


def test_swap_embed_rows_out_of_range_raises_valueerror():
    embed = torch.arange(20.0).reshape(5, 4)
    with pytest.raises(ValueError, match="DIŞINDA"):
        swap_embed_rows({"mapping.embed.weight": embed}, row_a=0, row_b=10)
