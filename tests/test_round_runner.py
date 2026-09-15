"""orchestrator/round_runner.py testleri.

`run_round`/`generate_and_submit_proof`/`main` gerçek pkl+ezkl+anvil+solc+
web3+kubo gerektirir (BİLİNÇLİ TEST SINIRI, sadece Colab'da
`tests/test_contracts.py` ile doğrulanır). Burada sadece saf/dosya-tabanlı
yardımcılar (`load_progress`/`save_progress`, `compute_weight_commitment`,
`load_circuit_config`) test edilir.
"""

import pytest
import torch

from orchestrator.round_runner import compute_weight_commitment, load_circuit_config, load_progress, save_progress
from storage.hashing import canonical_hash


def test_compute_weight_commitment_matches_canonical_hash_as_bytes():
    state = {"w": torch.randn(3, 3)}
    commitment = compute_weight_commitment(state)
    assert isinstance(commitment, bytes)
    assert len(commitment) == 32
    assert commitment == bytes.fromhex(canonical_hash(state))


def test_compute_weight_commitment_deterministic():
    state = {"w": torch.zeros(2, 2)}
    assert compute_weight_commitment(state) == compute_weight_commitment(state)


def test_load_progress_returns_empty_skeleton_when_missing(tmp_path):
    assert load_progress(str(tmp_path / "missing.json")) == {"rounds": {}}


def test_save_progress_then_load_progress_roundtrip(tmp_path):
    path = str(tmp_path / "progress.json")
    save_progress({"rounds": {"0": {"status": "done"}}}, path)
    assert load_progress(path) == {"rounds": {"0": {"status": "done"}}}
    assert not (tmp_path / "progress.json.tmp").exists()


def test_load_circuit_config_reads_real_file():
    config = load_circuit_config("configs/circuit.yaml")
    assert config["challenge_size_k"] == 1
    assert config["embed_mode"] == "matmul"
    assert config["scale"] == 8


def test_load_circuit_config_raises_on_missing_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("challenge_size_k: null\nembed_mode: matmul\nscale: 8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="eksik"):
        load_circuit_config(str(bad))
