import torch

import pytest

from fl.fedavg_utils import (
    RunningAverage,
    assert_matching_keys,
    compare_state_dicts,
    compute_norm_percentiles,
)


def test_running_average_matches_manual_mean():
    a = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([10.0])}
    b = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([20.0])}
    c = {"w": torch.tensor([5.0, 6.0]), "b": torch.tensor([30.0])}

    avg = RunningAverage()
    avg.update(a)
    avg.update(b)
    avg.update(c)
    result = avg.result()

    assert avg.count == 3
    assert torch.allclose(result["w"], torch.tensor([3.0, 4.0]))
    assert torch.allclose(result["b"], torch.tensor([20.0]))


def test_running_average_raises_on_key_mismatch():
    avg = RunningAverage()
    avg.update({"w": torch.tensor([1.0])})
    with pytest.raises(ValueError):
        avg.update({"different_key": torch.tensor([1.0])})


def test_running_average_result_before_update_raises():
    avg = RunningAverage()
    with pytest.raises(RuntimeError):
        avg.result()


def test_compare_state_dicts_identical_tensors():
    a = {"w": torch.tensor([1.0, 2.0, 3.0])}
    b = {"w": torch.tensor([1.0, 2.0, 3.0])}
    result = compare_state_dicts(a, b)

    assert result["per_key"]["w"]["max_abs_diff"] == pytest.approx(0.0)
    assert result["per_key"]["w"]["cosine_similarity"] == pytest.approx(1.0)
    assert result["only_in_a"] == []
    assert result["only_in_b"] == []


def test_compare_state_dicts_reports_key_mismatch():
    a = {"w": torch.tensor([1.0]), "extra_a": torch.tensor([1.0])}
    b = {"w": torch.tensor([1.0]), "extra_b": torch.tensor([1.0])}
    result = compare_state_dicts(a, b)

    assert result["only_in_a"] == ["extra_a"]
    assert result["only_in_b"] == ["extra_b"]


def test_compute_norm_percentiles_known_values():
    deltas = list(range(1, 101))  # 1..100
    result = compute_norm_percentiles([float(d) for d in deltas])

    assert result["n"] == 100
    assert result["p50"] == pytest.approx(50.5, abs=1.0)
    assert result["p99"] > result["p90"] > result["p50"]


def test_compute_norm_percentiles_empty_raises():
    with pytest.raises(ValueError):
        compute_norm_percentiles([])


def test_assert_matching_keys_passes_silently_when_equal():
    assert_matching_keys({"a", "b"}, {"b", "a"}, context="test")  # raise etmemeli


def test_assert_matching_keys_raises_with_missing_and_extra_details():
    with pytest.raises(ValueError) as exc_info:
        assert_matching_keys({"a", "b", "c"}, {"b", "c", "d"}, context="round 0 vs fedavg_0.pt")

    message = str(exc_info.value)
    assert "round 0 vs fedavg_0.pt" in message
    assert "'a'" in message
    assert "'d'" in message
