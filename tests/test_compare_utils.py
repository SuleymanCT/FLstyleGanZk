import pytest
import torch

from circuits.compare_utils import assert_num_ws_copies_identical, compute_comparison_stats


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


def test_assert_num_ws_copies_identical_passes_when_all_same():
    base = torch.randn(3, 5)
    w = base.unsqueeze(1).repeat(1, 16, 1)
    assert_num_ws_copies_identical(w)  # raise etmemeli


def test_assert_num_ws_copies_identical_raises_when_differing():
    w = torch.randn(3, 16, 5)  # tamamen rastgele, kopyalar aynı olmayacak
    with pytest.raises(RuntimeError, match="BİREBİR AYNI değil"):
        assert_num_ws_copies_identical(w)


def test_assert_num_ws_copies_identical_raises_on_wrong_dim():
    w = torch.randn(3, 5)  # 2 boyutlu, 3 bekleniyor
    with pytest.raises(RuntimeError, match="3 boyutlu"):
        assert_num_ws_copies_identical(w)
