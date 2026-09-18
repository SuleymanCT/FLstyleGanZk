"""eval/metrics.py'nin SAF/dosya-tabanlı fonksiyonlarının testleri —
StyleGAN-XL reposu/GPU/gerçek veri kümesi GEREKTİRMEZ. Colab'a bağımlı
kısımlar (run_official_metric, class_confusion_matrix, vb.) BİLİNÇLİ
TEST SINIRI içinde, burada test edilmiyor.
"""

import json

import numpy as np
import pytest
import torch

from eval.metrics import (
    _force_kwarg_wrapper,
    build_fixed_class_batch,
    compute_kid_from_features,
    is_cuda_oom_error,
    load_metric_options_from_training_options,
    to_uint8_images,
)

# --- load_metric_options_from_training_options ---


def _write_training_options(path, *, dataset_path, extra_fields=None):
    data = {
        "training_set_kwargs": {
            "class_name": "training.dataset.ImageFolderDataset",
            "path": dataset_path,
            "resolution": 256,
            "max_size": None,
            "use_labels": True,
            "xflip": False,
        },
        "num_gpus": 1,
        "metrics": ["fid50k_full", "kid50k_full"],
    }
    if extra_fields:
        data.update(extra_fields)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def test_load_metric_options_reads_real_fields(tmp_path):
    dataset_dir = tmp_path / "dr_dataset"
    dataset_dir.mkdir()
    opts_path = tmp_path / "training_options.json"
    _write_training_options(opts_path, dataset_path=str(dataset_dir))

    result = load_metric_options_from_training_options(str(opts_path))
    assert result["dataset_kwargs"]["path"] == str(dataset_dir)
    assert result["num_gpus"] == 1
    assert result["metrics"] == ["fid50k_full", "kid50k_full"]


def test_load_metric_options_raises_on_missing_required_fields(tmp_path):
    opts_path = tmp_path / "training_options.json"
    with open(opts_path, "w", encoding="utf-8") as f:
        json.dump({"num_gpus": 1}, f)
    with pytest.raises(ValueError, match="eksik alanlar"):
        load_metric_options_from_training_options(str(opts_path))


def test_load_metric_options_raises_when_dataset_path_missing(tmp_path):
    opts_path = tmp_path / "training_options.json"
    _write_training_options(opts_path, dataset_path=str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError):
        load_metric_options_from_training_options(str(opts_path))


def test_load_metric_options_applies_dataset_root_override(tmp_path):
    # Orijinal kayıtlı yol (başka bir makineden, artık geçersiz).
    original_stale_path = "/some/original/colab/session/dr_data.zip"
    opts_path = tmp_path / "training_options.json"
    _write_training_options(opts_path, dataset_path=original_stale_path)

    override_root = tmp_path / "new_root"
    override_root.mkdir()
    (override_root / "dr_data.zip").write_bytes(b"fake")

    result = load_metric_options_from_training_options(str(opts_path), dataset_root_override=str(override_root))
    assert result["dataset_kwargs"]["path"] == str(override_root / "dr_data.zip")


# --- compute_kid_from_features ---


def test_compute_kid_from_features_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 16))
    kid = compute_kid_from_features(features, features, num_subsets=20, max_subset_size=100, rng=np.random.default_rng(1))
    assert abs(kid) < 0.05


def test_compute_kid_from_features_large_for_well_separated_distributions():
    rng = np.random.default_rng(0)
    gen = rng.normal(loc=0.0, size=(300, 16))
    real = rng.normal(loc=50.0, size=(300, 16))
    kid = compute_kid_from_features(gen, real, num_subsets=20, max_subset_size=100, rng=np.random.default_rng(1))
    assert kid > 1.0


def test_compute_kid_from_features_deterministic_with_seeded_rng():
    rng = np.random.default_rng(0)
    gen = rng.normal(size=(50, 8))
    real = rng.normal(size=(50, 8))
    kid_a = compute_kid_from_features(gen, real, num_subsets=10, max_subset_size=30, rng=np.random.default_rng(7))
    kid_b = compute_kid_from_features(gen, real, num_subsets=10, max_subset_size=30, rng=np.random.default_rng(7))
    assert kid_a == kid_b


def test_compute_kid_from_features_raises_on_mismatched_feature_dims():
    with pytest.raises(ValueError, match="özellik boyutları"):
        compute_kid_from_features(np.zeros((10, 4)), np.zeros((10, 5)))


def test_compute_kid_from_features_raises_when_too_few_samples():
    with pytest.raises(ValueError, match="en az 2 örnek"):
        compute_kid_from_features(np.zeros((1, 4)), np.zeros((1, 4)))


# --- build_fixed_class_batch ---


def test_build_fixed_class_batch_produces_one_hot_c_for_every_row():
    z, c = build_fixed_class_batch(5, class_index=2, z_dim=8, c_dim=5, generator=torch.Generator().manual_seed(0))
    assert z.shape == (5, 8)
    assert c.shape == (5, 5)
    for row in c:
        assert torch.equal(row, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))


def test_build_fixed_class_batch_raises_when_class_index_out_of_range():
    with pytest.raises(ValueError, match="DIŞINDA"):
        build_fixed_class_batch(2, class_index=5, z_dim=8, c_dim=5, generator=torch.Generator().manual_seed(0))


# --- to_uint8_images ---


def test_to_uint8_images_matches_known_reference_formula():
    raw = torch.tensor([-1.0, 0.0, 1.0])
    result = to_uint8_images(raw)
    assert result.dtype == torch.uint8
    assert torch.equal(result, torch.tensor([0, 128, 255], dtype=torch.uint8))


def test_to_uint8_images_clamps_out_of_range_values():
    raw = torch.tensor([-10.0, 10.0])
    result = to_uint8_images(raw)
    assert result[0].item() == 0
    assert result[1].item() == 255


# --- _force_kwarg_wrapper (force_stylegan_ops_impl'in saf çekirdeği) ---
# Faz F'nin Colab koşumunda StyleGAN-XL'in CUDA custom op derlemesi
# ('bias_act_plugin') başarısız oldu; internal katman kodu 'impl='
# kwarg'ını hiç geçmediği için _init()'i yamalamak yerine bu
# sarmalayıcıyla HER çağrıda impl='ref' ZORLANIYOR.


def test_force_kwarg_wrapper_forces_kwarg_even_when_caller_passes_different_value():
    def original(x, impl="cuda"):
        return impl

    wrapped = _force_kwarg_wrapper(original, "impl", "ref")
    assert wrapped(1, impl="cuda") == "ref"


def test_force_kwarg_wrapper_forces_kwarg_when_caller_omits_it_entirely():
    def original(x, impl="cuda"):
        return impl

    wrapped = _force_kwarg_wrapper(original, "impl", "ref")
    assert wrapped(1) == "ref"


def test_force_kwarg_wrapper_passes_through_other_args_and_kwargs_unchanged():
    def original(x, y, impl="cuda"):
        return (x, y, impl)

    wrapped = _force_kwarg_wrapper(original, "impl", "ref")
    assert wrapped(3, y=4) == (3, 4, "ref")


# --- is_cuda_oom_error ---
# Faz F'nin gerçek Colab koşumunda 'ref' modunda upfirdn2d._upfirdn2d_ref
# icindeki F.pad 13.37 GiB istedi ve torch.OutOfMemoryError firlatti -
# generate_class_images bu hatayi yakalayip batch boyutunu yariya
# indirip yeniden deniyor, bunun icin hangi hatanin GERCEKTEN bir OOM
# oldugunu (PyTorch surumune gore iki farkli sekilde gelebiliyor)
# guvenilir sekilde tespit etmesi gerekiyor.


def test_is_cuda_oom_error_true_for_message_containing_out_of_memory():
    assert is_cuda_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 13.37 GiB")) is True


def test_is_cuda_oom_error_false_for_unrelated_runtime_error():
    assert is_cuda_oom_error(RuntimeError("size mismatch")) is False


def test_is_cuda_oom_error_false_for_unrelated_exception_type():
    assert is_cuda_oom_error(ValueError("out of memory")) is False


def test_is_cuda_oom_error_true_for_real_torch_oom_error_class_if_available():
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is None:
        pytest.skip("bu PyTorch sürümünde torch.cuda.OutOfMemoryError yok")
    assert is_cuda_oom_error(oom_cls("unrelated message")) is True
