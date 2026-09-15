"""Faz D kabul testi: `contracts/RoundManager.sol`, anvil üzerinde,
GERÇEK ezkl ispatlarıyla (sahte bytes DEĞİL).

Bu ortamda (yerel, Windows) `ezkl`/`anvil`/`solc` yoksa test SKIP olur —
hata değildir, `pytest` yine de yeşil kalır (CLAUDE.md madde 7/8).
Colab'da (hepsi kurulu) gerçekten çalışır.

**Tasarım notu — gerçek Drive/pkl verisi KULLANILMIYOR:** devre,
`tests/test_rebuild_mapping.py`'deki gibi küçük bir SENTETİK mapping
shard'ından kurulur (Faz C0-C3'te kanıtlanmış aynı `circuits.rebuild_mapping`/
`circuits.export_mapping`/`scripts.bench_circuit` fonksiyonları
kullanılarak) — bu, testi Drive bağımlılığından kurtarır (herhangi bir
ezkl+anvil+solc+web3 ortamında çalışabilir, sadece Colab'a özgü değil)
VE hızlandırır, ama yine de GERÇEK ezkl setup/prove/verify zincirinden
geçer — "sahte bytes" (hiç ezkl'den geçmemiş uydurma veriler) DEĞİL.

Pahalı kurulum (ezkl setup+prove, ~30-75s, Faz C3 ölçümüne göre) modül
başına BİR KEZ yapılır (`_shared_setup` fixture, `scope="module"`),
testler onu farklı RoundManager çağrı dizileriyle yeniden kullanır.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("ezkl", reason="ezkl kurulu değil — Faz D testi sadece Colab'da (ya da ezkl+anvil+solc+web3 kurulu bir ortamda) koşar.")

if shutil.which("anvil") is None:
    pytest.skip("anvil PATH'te yok — Faz D testi sadece Colab'da koşar.", allow_module_level=True)
if shutil.which("solc") is None and shutil.which("solc-select") is None:
    pytest.skip("solc PATH'te yok — Faz D testi sadece Colab'da koşar.", allow_module_level=True)

from chain.anvil import AnvilProcess  # noqa: E402
from chain.client import RoundManagerClient, Web3Client  # noqa: E402
from chain.solc import compile_with_fallback_strategies  # noqa: E402
from circuits.export_mapping import export_to_onnx  # noqa: E402
from circuits.rebuild_mapping import build_pruned_mapping_network  # noqa: E402
from circuits.toy_pipeline import compile_verifier_solidity, generate_solidity_verifier  # noqa: E402
from scripts.bench_circuit import build_multi_input_json, run_ezkl_pipeline, run_prove_and_verify, write_json_file  # noqa: E402

Z_DIM = 4
EMBED_NUM = 6
EMBED_DIM = 4
W_DIM = 5
C_DIM = 3
SCALE = 8

REPUTATION_INITIAL = 100
REPUTATION_PENALTY = 20
REPUTATION_BONUS = 1
REPUTATION_THRESHOLD = 50


def _make_fake_shard(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "mapping.embed.weight": torch.randn(EMBED_NUM, EMBED_DIM, generator=g),
        "mapping.embed_proj.weight": torch.randn(Z_DIM, EMBED_DIM, generator=g),
        "mapping.embed_proj.bias": torch.randn(Z_DIM, generator=g),
        "mapping.fc0.weight": torch.randn(W_DIM, 2 * Z_DIM, generator=g),
        "mapping.fc0.bias": torch.randn(W_DIM, generator=g),
        "mapping.fc1.weight": torch.randn(W_DIM, W_DIM, generator=g),
        "mapping.fc1.bias": torch.randn(W_DIM, generator=g),
    }


@pytest.fixture(scope="module")
def _shared_setup():
    """Pahalı ezkl+solc kurulumu — modül başına BİR KEZ. Gerçek bir
    proof.json + gerçek bir verifier bytecode + gerçek bir RoundManager
    ABI/bytecode üretir; anvil AYRI, testler arasında paylaşılan tek bir
    süreçte kalır (deploy edilen kontratlar test'ler arasında kalıcı
    olmalı)."""
    with tempfile.TemporaryDirectory(prefix="zk_test_contracts_") as tmp_dir:
        work_dir = Path(tmp_dir)

        shard = _make_fake_shard()
        model = build_pruned_mapping_network(shard, num_classes=C_DIM, embed_mode="matmul")
        model.eval()
        torch.manual_seed(1)
        z = torch.randn(1, Z_DIM)
        c = F.one_hot(torch.tensor([0]), num_classes=C_DIM).to(torch.float32)

        onnx_path = work_dir / "mapping.onnx"
        export_to_onnx(model, z, c, str(onnx_path), opset_version=13)

        input_json_path = work_dir / "input.json"
        write_json_file(build_multi_input_json(z, c), str(input_json_path))

        setup_result = run_ezkl_pipeline(onnx_path, input_json_path, work_dir, SCALE)
        prove_result = run_prove_and_verify(input_json_path, setup_result["paths"], work_dir)

        with open(prove_result["proof_path"], encoding="utf-8") as f:
            proof_json = json.load(f)
        proof_bytes = bytes.fromhex(proof_json["proof"].removeprefix("0x")) if isinstance(proof_json["proof"], str) else proof_json["proof"]
        public_inputs = proof_json.get("instances") or proof_json.get("public_inputs") or []

        sol_path, _abi_path = generate_solidity_verifier(setup_result["paths"], work_dir)
        compiled_verifier = compile_verifier_solidity(sol_path)
        assert not compiled_verifier["exceeds_eip170"], "test devresi (küçük, sentetik) EIP-170'i AŞMAMALI"

        compiled_round_manager = compile_with_fallback_strategies(Path("contracts/RoundManager.sol"))

        with AnvilProcess() as anvil:
            assert anvil.accounts, "anvil hesap listesi boş"
            owner_address, owner_key = anvil.accounts[0]
            site_address, site_key = anvil.accounts[1]
            other_site_address, other_site_key = anvil.accounts[2]

            deploy_client = Web3Client(anvil.rpc_url)
            verifier_address, _gas = deploy_client.deploy_bytecode(compiled_verifier["bytecode"], owner_key)

            round_manager_address, _gas = deploy_client.deploy_contract(
                compiled_round_manager["abi"],
                compiled_round_manager["bytecode"],
                (REPUTATION_INITIAL, REPUTATION_PENALTY, REPUTATION_BONUS, REPUTATION_THRESHOLD),
                owner_key,
            )

            rm = RoundManagerClient(anvil.rpc_url, round_manager_address, compiled_round_manager["abi"])
            rm.register_site(site_address, owner_key)
            rm.register_site(other_site_address, owner_key)

            yield {
                "rm": rm,
                "owner_key": owner_key,
                "site_address": site_address,
                "site_key": site_key,
                "other_site_address": other_site_address,
                "other_site_key": other_site_key,
                "verifier_address": verifier_address,
                "proof_bytes": proof_bytes,
                "public_inputs": public_inputs,
                "work_dir": work_dir,
                # accounts[3+]: BİLEREK register_site'a hiç verilmedi -
                # "kayıtsız site" testleri için AYNI zincirin gerçek bir hesabı.
                "unregistered_address": anvil.accounts[3][0],
                "unregistered_key": anvil.accounts[3][1],
            }


def test_happy_path_start_update_prove_finalize(_shared_setup):
    s = _shared_setup
    rm, owner_key, site, site_key = s["rm"], s["owner_key"], s["site_address"], s["site_key"]

    round_id = 100
    challenge_seed, start_gas = rm.start_round(round_id, "QmGlobalCID", (0).to_bytes(32, "big"), owner_key)
    assert start_gas > 0
    assert rm.get_challenge_seed(round_id) == challenge_seed

    update_gas = rm.submit_update(round_id, "QmUpdateCID", (1).to_bytes(32, "big"), site_key)
    assert update_gas > 0

    verified, proof_gas = rm.submit_proof(round_id, s["verifier_address"], s["proof_bytes"], s["public_inputs"], site_key)
    assert verified is True
    assert proof_gas > 0
    assert rm.get_reputation(site) == REPUTATION_INITIAL + REPUTATION_BONUS

    finalize_gas = rm.finalize_round(round_id, "QmAggregateCID", [site], owner_key)
    assert finalize_gas > 0
    info = rm.get_round_info(round_id)
    assert info["finalized"] is True
    assert info["aggregate_cid"] == "QmAggregateCID"


def test_invalid_proof_rejected_and_reputation_drops(_shared_setup):
    s = _shared_setup
    rm, owner_key, site, site_key = s["rm"], s["owner_key"], s["other_site_address"], s["other_site_key"]

    round_id = 101
    rm.start_round(round_id, "QmGlobalCID", (0).to_bytes(32, "big"), owner_key)
    rm.submit_update(round_id, "QmUpdateCID", (1).to_bytes(32, "big"), site_key)

    # GERÇEK bir ezkl proof'unu (rastgele bytes DEĞİL) bilerek bozuyoruz -
    # Halo2 ispatları tek bir bit değişikliğine bile kriptografik olarak
    # duyarlıdır, bu yüzden gerçek verifier bunu REDDEDECEKTİR.
    corrupted = bytearray(s["proof_bytes"])
    corrupted[0] ^= 0xFF

    reputation_before = rm.get_reputation(site)
    verified, gas = rm.submit_proof(round_id, s["verifier_address"], bytes(corrupted), s["public_inputs"], site_key)
    assert verified is False
    assert gas > 0
    assert rm.get_reputation(site) == reputation_before - REPUTATION_PENALTY


def test_site_excluded_after_reputation_drops_below_threshold(_shared_setup):
    s = _shared_setup
    rm, owner_key, site, site_key = s["rm"], s["owner_key"], s["other_site_address"], s["other_site_key"]

    corrupted = bytearray(s["proof_bytes"])
    corrupted[0] ^= 0xFF

    # Bu site önceki testte zaten bir kez cezalandırıldı (100 -> 80).
    # Eşiğin (50) altına düşene kadar tekrar tekrar geçersiz ispat gönder.
    round_id = 200
    while rm.is_site_eligible(site):
        round_id += 1
        rm.start_round(round_id, "QmGlobalCID", (0).to_bytes(32, "big"), owner_key)
        rm.submit_update(round_id, "QmUpdateCID", (1).to_bytes(32, "big"), site_key)
        rm.submit_proof(round_id, s["verifier_address"], bytes(corrupted), s["public_inputs"], site_key)

    assert rm.is_site_eligible(site) is False
    assert rm.get_reputation(site) < REPUTATION_THRESHOLD


def test_double_submit_update_rejected(_shared_setup):
    s = _shared_setup
    rm, owner_key, site, site_key = s["rm"], s["owner_key"], s["site_address"], s["site_key"]

    round_id = 300
    rm.start_round(round_id, "QmGlobalCID", (0).to_bytes(32, "big"), owner_key)
    rm.submit_update(round_id, "QmUpdateCID1", (1).to_bytes(32, "big"), site_key)
    with pytest.raises(RuntimeError, match="İşlem başarısız"):
        rm.submit_update(round_id, "QmUpdateCID2", (2).to_bytes(32, "big"), site_key)


def test_unregistered_site_rejected(_shared_setup):
    s = _shared_setup
    rm, owner_key, unregistered_key = s["rm"], s["owner_key"], s["unregistered_key"]

    round_id = 400
    rm.start_round(round_id, "QmGlobalCID", (0).to_bytes(32, "big"), owner_key)
    with pytest.raises(RuntimeError, match="İşlem başarısız"):
        rm.submit_update(round_id, "QmUpdateCID", (1).to_bytes(32, "big"), unregistered_key)


def test_unverified_site_cannot_be_included_in_finalize_round(_shared_setup):
    s = _shared_setup
    rm, owner_key, site, site_key = s["rm"], s["owner_key"], s["site_address"], s["site_key"]

    corrupted = bytearray(s["proof_bytes"])
    corrupted[0] ^= 0xFF

    round_id = 500
    rm.start_round(round_id, "QmGlobalCID", (0).to_bytes(32, "big"), owner_key)
    rm.submit_update(round_id, "QmUpdateCID", (1).to_bytes(32, "big"), site_key)
    verified, _gas = rm.submit_proof(round_id, s["verifier_address"], bytes(corrupted), s["public_inputs"], site_key)
    assert verified is False

    with pytest.raises(RuntimeError, match="İşlem başarısız"):
        rm.finalize_round(round_id, "QmAggregateCID", [site], owner_key)
