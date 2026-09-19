"""attacks/ modülünün SAF (state_dict-only) fonksiyonlarının testleri —
hiçbiri gerçek StyleGAN-XL/pkl/GPU gerektirmez, tamamen sentetik
tensörlerle yerelde GERÇEKTEN test edilir."""

import pytest
import torch

from attacks.conditional_poison import compensated_swap_embed_rows, swap_embed_rows
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


# --- conditional_poison.compensated_swap_embed_rows ---
# Gerçek Colab bulgusu (H1 testi DOĞRULANDI): basit swap_embed_rows
# zehirli site TEK BAŞINA ölçüldüğünde takas GÖRÜNÜYOR ama 4 siteli
# FedAvg'dan SONRA sinyal seyreliyor/kayboluyor - compensated versiyon
# bunu formülle (num_sites*target - (num_sites-1)*honest) telafi ediyor.


def test_compensated_swap_embed_rows_matches_manual_formula():
    embed = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    state = {"mapping.embed.weight": embed}
    result = compensated_swap_embed_rows(state, row_a=0, row_b=1, num_sites=4)
    # row_a: 4*honest_b - 3*honest_a = 4*[2,2] - 3*[1,1] = [5,5]
    # row_b: 4*honest_a - 3*honest_b = 4*[1,1] - 3*[2,2] = [-2,-2]
    assert torch.allclose(result["mapping.embed.weight"][0], torch.tensor([5.0, 5.0]))
    assert torch.allclose(result["mapping.embed.weight"][1], torch.tensor([-2.0, -2.0]))
    # dokunulmayan satır aynen kalmalı
    assert torch.allclose(result["mapping.embed.weight"][2], torch.tensor([3.0, 3.0]))


def test_compensated_swap_embed_rows_averaged_with_honest_majority_reaches_exact_target():
    # FedAvg SIMULASYONU: 3 durust site (degismemis) + 1 zehirli site
    # (compensated). Ortalama TAM OLARAK hedefe (honest_row_b) ulasmali.
    honest_a, honest_b = torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])
    embed = torch.stack([honest_a, honest_b, torch.tensor([9.0, 9.0])])
    state = {"mapping.embed.weight": embed.clone()}
    result = compensated_swap_embed_rows(state, row_a=0, row_b=1, num_sites=4)
    poisoned_row_a = result["mapping.embed.weight"][0]

    fedavg_row_a = (3 * honest_a + poisoned_row_a) / 4
    assert torch.allclose(fedavg_row_a, honest_b, atol=1e-5)


def test_compensated_swap_embed_rows_delta_is_num_sites_times_simple_swap_delta():
    embed = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
    state = {"mapping.embed.weight": embed.clone()}
    simple = swap_embed_rows(state, row_a=0, row_b=1)
    compensated = compensated_swap_embed_rows(state, row_a=0, row_b=1, num_sites=4)

    simple_delta = simple["mapping.embed.weight"] - embed
    compensated_delta = compensated["mapping.embed.weight"] - embed
    assert torch.allclose(compensated_delta, 4 * simple_delta, atol=1e-5)


def test_compensated_swap_embed_rows_leaves_original_dict_untouched():
    embed = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    original = embed.clone()
    state = {"mapping.embed.weight": embed}
    compensated_swap_embed_rows(state, row_a=0, row_b=1, num_sites=4)
    assert torch.equal(state["mapping.embed.weight"], original)


def test_compensated_swap_embed_rows_raises_on_num_sites_below_two():
    embed = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="num_sites en az 2"):
        compensated_swap_embed_rows({"mapping.embed.weight": embed}, row_a=0, row_b=1, num_sites=1)


def test_compensated_swap_embed_rows_missing_key_raises_keyerror():
    with pytest.raises(KeyError):
        compensated_swap_embed_rows({"other.key": torch.ones(3)}, row_a=0, row_b=1, num_sites=4)
