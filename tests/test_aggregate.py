"""orchestrator/aggregate.py testleri.

En kritik test `test_aggregate_round_matches_running_average_when_nothing_excluded`:
`aggregate_round`'un kapı mantığı hiçbir şeyi dışlamadığında, sonucun
`fl.fedavg_utils.RunningAverage`'in DOĞRUDAN çağrılmasıyla BİREBİR AYNI
olduğunu kanıtlar. `scripts/audit_fedavg.py` da aynı `RunningAverage`
sınıfını kullandığından (bkz. fl/fedavg_utils.py docstring'i), bu iki
kullanım noktası TANIM GEREĞİ aynı aritmetiği paylaşır — bu test o
garantinin `aggregate_round`'un EK kapı mantığı tarafından bozulmadığını
doğruluyor.
"""

import pytest
import torch

from fl.fedavg_utils import RunningAverage
from orchestrator.aggregate import aggregate_round, compute_delta_norm, gate_site_update, has_nan_or_inf

TAU = 3000.0


def _state(seed: int, scale: float = 1.0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(4, 4, generator=g) * scale, "b": torch.randn(4, generator=g) * scale}


def test_compute_delta_norm_known_value():
    prev = {"w": torch.zeros(3)}
    new = {"w": torch.tensor([3.0, 4.0, 0.0])}
    assert compute_delta_norm(prev, new) == pytest.approx(5.0)


def test_compute_delta_norm_rejects_mismatched_keys():
    with pytest.raises(ValueError, match="anahtar"):
        compute_delta_norm({"a": torch.zeros(1)}, {"b": torch.zeros(1)})


def test_has_nan_or_inf_detects_nan():
    assert has_nan_or_inf({"w": torch.tensor([1.0, float("nan")])}) is True


def test_has_nan_or_inf_detects_inf():
    assert has_nan_or_inf({"w": torch.tensor([1.0, float("inf")])}) is True


def test_has_nan_or_inf_false_for_clean_tensor():
    assert has_nan_or_inf({"w": torch.tensor([1.0, 2.0])}) is False


def test_gate_site_update_excludes_when_not_chain_approved():
    prev = _state(1)
    gate = gate_site_update("s1", prev, _state(2), tau_norm_threshold=TAU, chain_approved=False)
    assert gate["included"] is False
    assert "zincir onayı yok" in gate["reasons"][0]


def test_gate_site_update_excludes_on_norm_violation():
    prev = {"w": torch.zeros(10)}
    huge_update = {"w": torch.ones(10) * 10_000}
    gate = gate_site_update("s1", prev, huge_update, tau_norm_threshold=TAU, chain_approved=True)
    assert gate["included"] is False
    assert gate["delta_norm"] > TAU


def test_gate_site_update_excludes_on_nan():
    prev = {"w": torch.zeros(3)}
    bad = {"w": torch.tensor([1.0, float("nan"), 2.0])}
    gate = gate_site_update("s1", prev, bad, tau_norm_threshold=TAU, chain_approved=True)
    assert gate["included"] is False
    assert "NaN/Inf" in gate["reasons"][0]


def test_gate_site_update_includes_clean_approved_update():
    prev = _state(1)
    gate = gate_site_update("s1", prev, _state(2, scale=0.01), tau_norm_threshold=TAU, chain_approved=True)
    assert gate["included"] is True
    assert gate["reasons"] == []


def test_aggregate_round_raises_on_empty_site_states():
    with pytest.raises(ValueError, match="site_states boş"):
        aggregate_round({}, {}, tau_norm_threshold=TAU, chain_approved={})


def test_aggregate_round_matches_running_average_when_nothing_excluded():
    prev = _state(0)
    site_states = {"s1": _state(1, scale=0.01), "s2": _state(2, scale=0.01), "s3": _state(3, scale=0.01)}
    chain_approved = {"s1": True, "s2": True, "s3": True}

    result = aggregate_round(prev, site_states, tau_norm_threshold=TAU, chain_approved=chain_approved)

    running = RunningAverage()
    for state in site_states.values():
        running.update(state)
    expected = running.result()

    assert result["num_included"] == 3
    assert result["excluded_sites"] == []
    for key in expected:
        assert torch.equal(result["global_state"][key], expected[key])


def test_aggregate_round_excludes_unapproved_site_and_reports_activation_rate():
    prev = _state(0)
    site_states = {"s1": _state(1, scale=0.01), "s2": _state(2, scale=0.01)}
    chain_approved = {"s1": True, "s2": False}

    result = aggregate_round(prev, site_states, tau_norm_threshold=TAU, chain_approved=chain_approved)

    assert result["included_sites"] == ["s1"]
    assert result["excluded_sites"] == ["s2"]
    assert result["activation_rate"] == pytest.approx(0.5)

    running = RunningAverage()
    running.update(site_states["s1"])
    expected = running.result()
    for key in expected:
        assert torch.equal(result["global_state"][key], expected[key])


def test_aggregate_round_raises_when_all_sites_excluded():
    prev = _state(0)
    site_states = {"s1": _state(1)}
    with pytest.raises(RuntimeError, match="Hiçbir site"):
        aggregate_round(prev, site_states, tau_norm_threshold=TAU, chain_approved={"s1": False})
