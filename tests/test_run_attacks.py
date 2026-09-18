"""scripts/run_attacks.py'nin SAF fonksiyonlarının testleri —
StyleGAN-XL/GPU/gerçek veri kümesi GEREKTİRMEZ. `main()`/`evaluate_condition`
(GPU+gerçek checkpoint+dataset gerektirir) BİLİNÇLİ TEST SINIRI içinde,
burada test edilmiyor.
"""

import os

import pytest
import torch

from scripts.run_attacks import ATTACK_NAMES, apply_attack, resolve_training_options_path, result_key


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
