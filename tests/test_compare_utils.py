import pytest
import torch

from circuits.compare_utils import compute_comparison_stats


def test_identical_tensors_zero_diff_full_cosine():
    a = torch.tensor([1.0, 2.0, 3.0])
    stats = compute_comparison_stats(a, a.clone())
    assert stats["max_abs_diff"] == pytest.approx(0.0)
    assert stats["mean_abs_diff"] == pytest.approx(0.0)
    assert stats["cosine_similarity"] == pytest.approx(1.0)


def test_known_difference():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 2.0, 3.0, 8.0])
    stats = compute_comparison_stats(a, b)
    assert stats["max_abs_diff"] == pytest.approx(4.0)
    assert stats["mean_abs_diff"] == pytest.approx(1.0)
    assert 0.9 < stats["cosine_similarity"] < 1.0


def test_shape_mismatch_raises():
    a = torch.zeros(3)
    b = torch.zeros(4)
    with pytest.raises(ValueError):
        compute_comparison_stats(a, b)


def test_opposite_vectors_negative_cosine():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([-1.0, 0.0])
    stats = compute_comparison_stats(a, b)
    assert stats["cosine_similarity"] == pytest.approx(-1.0)
