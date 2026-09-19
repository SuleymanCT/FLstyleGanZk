"""scripts/run_attacks.py'nin SAF fonksiyonlarının testleri —
StyleGAN-XL/GPU/gerçek veri kümesi GEREKTİRMEZ. `main()`/`evaluate_condition`
(GPU+gerçek checkpoint+dataset gerektirir) BİLİNÇLİ TEST SINIRI içinde,
burada test edilmiyor.
"""

import os

import pytest
import torch

from scripts.run_attacks import (
    ATTACK_NAMES,
    CONDITIONS,
    apply_attack,
    build_poisoned_alone_global,
    estimate_full_mode_cost,
    parse_args,
    pick_default_gen_batch_size,
    resolve_training_options_path,
    result_key,
)


def _synthetic_site_and_prev():
    site = {
        "mapping.embed.weight": torch.arange(20.0).reshape(5, 4),
        "mapping.fc0.weight": torch.tensor([[5.0, 5.0]]),
    }
    prev = {
        "mapping.embed.weight": torch.zeros(5, 4),
        "mapping.fc0.weight": torch.tensor([[1.0, 1.0]]),
    }
    return site, prev


def test_result_key_format():
    assert result_key("random_weights", "unprotected") == "random_weights__unprotected"


def test_apply_attack_random_weights_changes_all_keys_and_reports_them():
    site, prev = _synthetic_site_and_prev()
    poisoned, modified_keys = apply_attack("random_weights", poisoned_site_state=site, prev_global_state=prev)
    assert set(modified_keys) == set(site.keys())
    assert not torch.equal(poisoned["mapping.embed.weight"], site["mapping.embed.weight"])


def test_apply_attack_scaled_poison_matches_manual_arithmetic():
    site, prev = _synthetic_site_and_prev()
    poisoned, modified_keys = apply_attack("scaled_poison_10x", poisoned_site_state=site, prev_global_state=prev)
    # fc0: prev=1, site=5, delta=4 -> poisoned = 1 + 10*4 = 41
    assert torch.allclose(poisoned["mapping.fc0.weight"], torch.tensor([[41.0, 41.0]]))
    assert set(modified_keys) == set(site.keys())


def test_apply_attack_scaled_poison_variants_use_correct_scale():
    site, prev = _synthetic_site_and_prev()
    p50, _ = apply_attack("scaled_poison_50x", poisoned_site_state=site, prev_global_state=prev)
    p100, _ = apply_attack("scaled_poison_100x", poisoned_site_state=site, prev_global_state=prev)
    # fc0: delta=4 -> 1+50*4=201, 1+100*4=401
    assert torch.allclose(p50["mapping.fc0.weight"], torch.tensor([[201.0, 201.0]]))
    assert torch.allclose(p100["mapping.fc0.weight"], torch.tensor([[401.0, 401.0]]))


def test_apply_attack_conditional_poison_only_reports_embed_key():
    site, prev = _synthetic_site_and_prev()
    poisoned, modified_keys = apply_attack("conditional_poison", poisoned_site_state=site, prev_global_state=prev)
    assert modified_keys == ["mapping.embed.weight"]
    assert torch.equal(poisoned["mapping.embed.weight"][0], site["mapping.embed.weight"][4])
    assert torch.equal(poisoned["mapping.embed.weight"][4], site["mapping.embed.weight"][0])
    # fc0 saldırıdan ETKİLENMEMELİ
    assert torch.equal(poisoned["mapping.fc0.weight"], site["mapping.fc0.weight"])


def test_apply_attack_conditional_poison_compensated_reports_only_embed_key():
    site, prev = _synthetic_site_and_prev()
    poisoned, modified_keys = apply_attack("conditional_poison_compensated", poisoned_site_state=site, prev_global_state=prev)
    assert modified_keys == ["mapping.embed.weight"]
    # fc0 saldırıdan ETKİLENMEMELİ
    assert torch.equal(poisoned["mapping.fc0.weight"], site["mapping.fc0.weight"])


def test_apply_attack_compensated_embed_delta_is_num_sites_times_simple_swap_delta():
    # Kompanzasyonun ||ΔG||'yi buyutme miktari: embed delta'si tam NUM_SITES kat.
    from scripts.run_attacks import NUM_SITES

    site, prev = _synthetic_site_and_prev()
    simple, _ = apply_attack("conditional_poison", poisoned_site_state=site, prev_global_state=prev)
    comp, _ = apply_attack("conditional_poison_compensated", poisoned_site_state=site, prev_global_state=prev)
    simple_delta = simple["mapping.embed.weight"] - site["mapping.embed.weight"]
    comp_delta = comp["mapping.embed.weight"] - site["mapping.embed.weight"]
    assert torch.allclose(comp_delta, NUM_SITES * simple_delta, atol=1e-4)


def test_apply_attack_unknown_name_raises():
    site, prev = _synthetic_site_and_prev()
    with pytest.raises(ValueError, match="Bilinmeyen saldırı"):
        apply_attack("not_a_real_attack", poisoned_site_state=site, prev_global_state=prev)


def test_all_declared_attack_names_are_handled_by_apply_attack():
    site, prev = _synthetic_site_and_prev()
    for name in ATTACK_NAMES:
        poisoned, modified_keys = apply_attack(name, poisoned_site_state=site, prev_global_state=prev)
        assert isinstance(poisoned, dict) and modified_keys


# --- resolve_training_options_path (gerçek pkl gerekmeden, sentetik dizin ağacı) ---


def test_resolve_training_options_path_finds_real_file(tmp_path):
    site_dir = tmp_path / "round_10" / "site_0"
    run_dir = site_dir / "00000-stylegan3-r-site0"
    run_dir.mkdir(parents=True)
    (run_dir / "training_options.json").write_text("{}", encoding="utf-8")

    result = resolve_training_options_path(str(tmp_path), 10, 0)
    assert result == str(run_dir / "training_options.json")


def test_resolve_training_options_path_missing_site_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_training_options_path(str(tmp_path), 10, 0)


def test_resolve_training_options_path_missing_file_raises(tmp_path):
    run_dir = tmp_path / "round_10" / "site_0" / "00000-run"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        resolve_training_options_path(str(tmp_path), 10, 0)


# --- estimate_full_mode_cost (ref moda düşüldüğünde --metrics-mode full
# ekstrapolasyonu — Faz F'nin gerçek Colab koşumunda CUDA custom op
# derlemesi başarısız oldu, bu tahmin GERÇEK ölçülen seconds_per_image'dan
# hesaplanıyor, uydurulmuyor) ---


def test_estimate_full_mode_cost_matches_manual_arithmetic():
    # 1 saniye/görüntü, 50000 görüntü/metrik * 2 metrik = 100000s/koşul = ~27.78 saat
    result = estimate_full_mode_cost(1.0, num_conditions=10, images_per_metric_run=50000, metrics_per_condition=2)
    assert result["hours_per_condition"] == pytest.approx(100000 / 3600)
    assert result["hours_total"] == pytest.approx(100000 / 3600 * 10)
    assert result["num_conditions"] == 10


def test_estimate_full_mode_cost_scales_linearly_with_seconds_per_image():
    slow = estimate_full_mode_cost(2.0, num_conditions=5)
    fast = estimate_full_mode_cost(1.0, num_conditions=5)
    assert slow["hours_total"] == pytest.approx(fast["hours_total"] * 2)


def test_estimate_full_mode_cost_raises_on_non_positive_input():
    with pytest.raises(ValueError, match="pozitif"):
        estimate_full_mode_cost(0.0)
    with pytest.raises(ValueError, match="pozitif"):
        estimate_full_mode_cost(-1.0)


# --- pick_default_gen_batch_size ---
# Faz F'nin gerçek Colab koşumunda 'ref' modunda upfirdn2d'nin saf
# PyTorch yolu TEK bir görüntü için bile CUDA OOM verdi (13.37 GiB) -
# bu yüzden 'ref' için varsayılan batch boyutu KÜÇÜK olmalı.


def test_pick_default_gen_batch_size_ref_is_small():
    assert pick_default_gen_batch_size("ref") == 1


def test_pick_default_gen_batch_size_cuda_is_larger():
    assert pick_default_gen_batch_size("cuda") > pick_default_gen_batch_size("ref")


def test_pick_default_gen_batch_size_raises_on_unknown_ops_impl():
    with pytest.raises(ValueError, match="Bilinmeyen ops_impl"):
        pick_default_gen_batch_size("not_a_real_impl")


# --- --device / --gen-batch-size CLI argümanları ---


def test_parse_args_device_defaults_to_auto():
    args = parse_args([])
    assert args.device == "auto"
    assert args.gen_batch_size is None


def test_parse_args_accepts_device_and_gen_batch_size_overrides():
    args = parse_args(["--device", "cpu", "--gen-batch-size", "2"])
    assert args.device == "cpu"
    assert args.gen_batch_size == 2


def test_parse_args_accepts_custom_metrics_mode_and_fid_num_gen():
    # Faz F'nin gercek Colab kosumunda 'ref' modunda --metrics-mode full
    # ~138 saat surdugunden pratik degildi - 'custom' modu bunun icin
    # eklendi (13.13 ile karsilastirilamaz ama GERCEK bir olcum).
    args = parse_args(["--metrics-mode", "custom", "--fid-num-gen", "3000"])
    assert args.metrics_mode == "custom"
    assert args.fid_num_gen == 3000


def test_parse_args_fid_num_gen_defaults_to_none():
    args = parse_args([])
    assert args.fid_num_gen is None


# --- build_poisoned_alone_global (H1 tanı koşulu) ---
# Gerçek Colab bulgusu: conditional_poison'ın DR-0/DR-4 takası FedAvg'lı
# (4 site) sonuç matrisinde GÖRÜNMEDİ - H1 hipotezi bunun 3 dürüst
# site tarafından SEYRELTİLMESİNDEN kaynaklanıp kaynaklanmadığını,
# FedAvg'ı TAMAMEN ATLAYIP zehirli site'ı TEK BAŞINA izole ederek
# test ediyor.


def test_conditions_includes_poisoned_alone():
    assert "poisoned_alone" in CONDITIONS


def test_build_poisoned_alone_global_uses_poisoned_state_directly_no_averaging():
    poisoned_state = {"mapping.embed.weight": torch.ones(5, 4) * 99.0}
    result = build_poisoned_alone_global(poisoned_state, poisoned_site=2)
    assert result["global_state"] is poisoned_state
    assert torch.equal(result["global_state"]["mapping.embed.weight"], poisoned_state["mapping.embed.weight"])


def test_build_poisoned_alone_global_only_includes_the_poisoned_site():
    result = build_poisoned_alone_global({"a": torch.zeros(1)}, poisoned_site=3)
    assert result["included_sites"] == [3]
