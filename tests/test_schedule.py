import pytest

from orchestrator.schedule import (
    build_round_schedule,
    deterministic_unit_interval,
    load_schedule_config,
    must_prove,
)

SEED_A = (1).to_bytes(32, "big")
SEED_B = (2).to_bytes(32, "big")


def test_load_schedule_config_reads_real_file():
    config = load_schedule_config("configs/schedule.yaml")
    assert config["reputation_initial"] == 100
    assert config["reputation_penalty"] == 20
    assert config["reputation_bonus"] == 1
    assert config["tau_norm_threshold"] == 3000.0
    assert config["proof_schedule"]["fixed_rounds"] == [0, 7, 14]
    assert config["proof_schedule"]["reputation_threshold"] == 50


def test_load_schedule_config_raises_on_missing_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("proof_schedule: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="eksik anahtarlar"):
        load_schedule_config(str(bad))


def test_deterministic_unit_interval_is_deterministic_and_bounded():
    v1 = deterministic_unit_interval(SEED_A, 3, "0xSiteA")
    v2 = deterministic_unit_interval(SEED_A, 3, "0xSiteA")
    assert v1 == v2
    assert 0.0 <= v1 < 1.0


def test_deterministic_unit_interval_case_insensitive_site():
    assert deterministic_unit_interval(SEED_A, 3, "0xAbC") == deterministic_unit_interval(SEED_A, 3, "0xabc")


def test_deterministic_unit_interval_varies_by_site():
    assert deterministic_unit_interval(SEED_A, 3, "0xSiteA") != deterministic_unit_interval(SEED_A, 3, "0xSiteB")


def test_deterministic_unit_interval_accepts_int_site_without_crashing():
    # Faz E'nin gerçek Colab koşumunda tam BURADA çöktü:
    # replay_proofs.py site'ı int (anvil hesap indeksi) olarak geçiriyordu,
    # eski kod sadece str varsayıp .lower() çağırıyordu -> AttributeError.
    value = deterministic_unit_interval(SEED_A, 3, 2)
    assert 0.0 <= value < 1.0


def test_deterministic_unit_interval_int_and_str_site_give_same_result():
    assert deterministic_unit_interval(SEED_A, 3, 2) == deterministic_unit_interval(SEED_A, 3, "2")


def test_deterministic_unit_interval_same_triple_always_matches():
    # (seed, round, site) üçlüsü aynıysa değer HER ZAMAN aynı olmalı -
    # hem tekrarlı çağrıda hem farklı tiplerle (int/str) ifade edildiğinde.
    for _ in range(5):
        assert deterministic_unit_interval(SEED_A, 7, 1) == deterministic_unit_interval(SEED_A, 7, "1")
    assert deterministic_unit_interval(SEED_A, 7, 1) != deterministic_unit_interval(SEED_B, 7, 1)
    assert deterministic_unit_interval(SEED_A, 7, 1) != deterministic_unit_interval(SEED_A, 8, 1)


def test_must_prove_full_mode_always_true():
    assert must_prove(1, "s", challenge_seed=SEED_A, reputation=100, reputation_threshold=50,
                       fixed_rounds=[], random_ratio=0.0, mode="full") is True


def test_must_prove_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Bilinmeyen mod"):
        must_prove(1, "s", challenge_seed=SEED_A, reputation=100, reputation_threshold=50,
                   fixed_rounds=[], random_ratio=0.0, mode="bogus")


def test_must_prove_fixed_round_forces_true_regardless_of_reputation_and_ratio():
    assert must_prove(7, "s", challenge_seed=SEED_A, reputation=100, reputation_threshold=50,
                       fixed_rounds=[7], random_ratio=0.0, mode="sampled") is True


def test_must_prove_low_reputation_forces_true():
    assert must_prove(3, "s", challenge_seed=SEED_A, reputation=10, reputation_threshold=50,
                       fixed_rounds=[], random_ratio=0.0, mode="sampled") is True


def test_must_prove_high_reputation_non_fixed_round_falls_back_to_random_ratio():
    # random_ratio=0.0 -> asla secilmez
    assert must_prove(3, "s", challenge_seed=SEED_A, reputation=100, reputation_threshold=50,
                       fixed_rounds=[], random_ratio=0.0, mode="sampled") is False
    # random_ratio=1.0 -> her zaman secilir
    assert must_prove(3, "s", challenge_seed=SEED_A, reputation=100, reputation_threshold=50,
                       fixed_rounds=[], random_ratio=1.0, mode="sampled") is True


def test_must_prove_null_random_ratio_treated_as_never_random_selected():
    assert must_prove(3, "s", challenge_seed=SEED_A, reputation=100, reputation_threshold=50,
                       fixed_rounds=[], random_ratio=None, mode="sampled") is False


def test_build_round_schedule_combines_all_sites():
    config = {
        "proof_schedule": {"fixed_rounds": [0], "random_ratio": 0.0, "reputation_threshold": 50},
        "reputation_initial": 100,
    }
    schedule = build_round_schedule(
        0, ["siteA", "siteB"], challenge_seed=SEED_A, reputations={"siteA": 100, "siteB": 10}, config=config
    )
    # round 0 fixed -> hepsi True
    assert schedule == {"siteA": True, "siteB": True}

    schedule_non_fixed = build_round_schedule(
        1, ["siteA", "siteB"], challenge_seed=SEED_A, reputations={"siteA": 100, "siteB": 10}, config=config
    )
    # round 1 fixed degil, random_ratio=0.0 -> sadece dusuk itibarli (siteB) True
    assert schedule_non_fixed == {"siteA": False, "siteB": True}


def test_build_round_schedule_uses_reputation_initial_for_unknown_site():
    config = {
        "proof_schedule": {"fixed_rounds": [], "random_ratio": 0.0, "reputation_threshold": 50},
        "reputation_initial": 100,
    }
    schedule = build_round_schedule(1, ["newSite"], challenge_seed=SEED_A, reputations={}, config=config)
    assert schedule == {"newSite": False}
