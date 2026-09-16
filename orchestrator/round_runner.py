#!/usr/bin/env python
"""Faz D: bir round'un uçtan uca akışı (ince orkestrasyon).

**Bu bir CANLI eğitim turu BAŞLATMIYOR** — CLAUDE.md madde 2/3 gereği
("Base model yeniden eğitilmez. 15 round yeniden eğitilmez." /
"StyleGAN-XL eğitim döngüsünün matematiğine dokunulmaz.") her site'ın
"yerel eğitim" adımı Faz A'da ZATEN üretilmiş snapshot'ın REPLAY'i
(`fl.round_replay.load_site_update`) — bu script mevcut bir round'u
protokol açısından (challenge/ispat/agregasyon/finalize) simüle ediyor.
Gerçek canlı entegre koşu Faz F'in kapsamı.

Akış (her round için):
  1. `startRound` -> zincirden `challengeSeed` alınır.
  2. `challengeSeed`'den deterministik (z,c) üretilir (k=1).
  3. Her site için: önceki round'un global G'si (round_replay ile) yüklenir,
     `globalHash` ile karşılaştırılır; site'ın YENİ (replay) G_ema'sı yüklenir.
  4. Ağırlık taahhüdü (`canonical_hash`) hesaplanır, güncelleme yerel
     `work_dir`'e kaydedilip IPFS'e yüklenir (`updateCID`), `submitUpdate`
     çağrılır.
  5. `orchestrator.schedule.must_prove` bu (round,site)'ın ispat üretip
     üretmeyeceğine karar verir.
  6. Üretecekse: mapping shard'ı budanır (embed_mode=matmul, `configs/circuit.yaml`),
     ONNX'e ihraç edilir, TAM ezkl pipeline'ı (`scripts.bench_circuit.run_ezkl_pipeline`/
     `run_prove_and_verify`) çalıştırılır, BU SİTEYE ÖZGÜ bir Verifier
     deploy edilir (bkz. `contracts/RoundManager.sol`'un "ÖNEMLİ TASARIM
     NOTU" — `param_visibility="fixed"` yüzünden her (round,site)'ın
     KENDİ verifier'ı olur), `submitProof` çağrılır.
  7. Zincirden her site'ın onay durumu (`getSubmission`/`isSiteEligible`)
     okunur.
  8. `orchestrator.aggregate.aggregate_round` ile fp32 FedAvg (SADECE
     onaylı site'lar) hesaplanır.
  9. Agregasyon sonucu IPFS'e yüklenir (`aggregateCID`), `finalizeRound`
     çağrılır.

`progress.json` ile devam edilebilir (Colab kopmalarına karşı) — her
tamamlanan (round,site) adımı işaretlenip diske yazılır, yeniden
çalıştırıldığında zaten tamamlanmış adımlar ATLANIR.

BİLİNÇLİ TEST SINIRI: raw_root'taki gerçek pkl'ler, StyleGAN-XL reposu,
ezkl, anvil, solc, web3, kubo IPFS node'u gerektirir — SADECE Colab'da
çalışır/test edilir (`tests/test_contracts.py`). Saf yardımcılar
(`load_progress`/`save_progress`, `compute_weight_commitment`)
`tests/test_round_runner.py`'de yerelde test edilir.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml

from orchestrator.aggregate import aggregate_round
from orchestrator.challenge import build_challenge_z_c
from orchestrator.schedule import build_round_schedule, load_schedule_config
from scripts.bench_circuit import build_multi_input_json, default_work_root, write_json_file
from storage.hashing import canonical_hash
from storage.pathguard import assert_writable

DEFAULT_CIRCUIT_CONFIG = "configs/circuit.yaml"
MAPPING_PREFIXES = ["mapping"]


def load_circuit_config(config_path: str = DEFAULT_CIRCUIT_CONFIG) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    required = ("challenge_size_k", "embed_mode", "scale")
    missing = [k for k in required if data.get(k) is None]
    if missing:
        raise ValueError(f"'{config_path}' içinde eksik/null alanlar: {missing}")
    return data


def compute_weight_commitment(state_dict: dict) -> bytes:
    """`canonical_hash` (hex sha256, 64 karakter) -> `bytes32` (32 byte) —
    `RoundManager.sol`'un `weightCommitment`/`globalHash` alanlarına
    doğrudan verilebilir biçim."""
    return bytes.fromhex(canonical_hash(state_dict))


def load_progress(path: str) -> dict:
    if not os.path.isfile(path):
        return {"rounds": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_progress(progress: dict, path: str) -> None:
    assert_writable(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    assert_writable(tmp_path)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def upload_state_dict_to_ipfs(state_dict: dict, work_dir: Path, filename: str) -> str:
    from storage.ipfs import add

    local_path = work_dir / filename
    torch.save(state_dict, local_path)
    return add(str(local_path))


def generate_and_submit_proof(
    *,
    round_id: int,
    site: str,
    mapping_shard: dict,
    z: torch.Tensor,
    c: torch.Tensor,
    circuit_config: dict,
    work_dir: Path,
    round_manager_client,
    private_key: str,
) -> dict:
    """Bu (round,site)'a ÖZGÜ bir devre kurar (site'ın GÜNCEL ağırlıkları
    `param_visibility="fixed"` ile devrenin sabit sütunlarına gömülür),
    TAM ezkl pipeline'ını çalıştırıp KENDİ Verifier'ını deploy eder ve
    `submitProof`'u çağırır. Hiçbir ezkl/solc adımı burada yeniden
    yazılmıyor — `scripts.bench_circuit`/`circuits.toy_pipeline` (Faz B/C3'te
    kanıtlanmış) fonksiyonları DOĞRUDAN yeniden kullanılıyor."""
    import ezkl  # noqa: F401 - erken/net "ezkl kurulu değil" hatası için
    from chain.client import Web3Client
    from chain.solc import EIP170_MAX_DEPLOYED_BYTECODE_SIZE
    from circuits.export_mapping import export_to_onnx
    from circuits.ezkl_utils import parse_proof_bytes, parse_public_inputs
    from circuits.rebuild_mapping import build_pruned_mapping_network, verify_prunable_embed_rows
    from circuits.toy_pipeline import compile_verifier_solidity, generate_solidity_verifier
    from scripts.bench_circuit import run_ezkl_pipeline, run_prove_and_verify

    c_dim = c.shape[1]
    verify_prunable_embed_rows([c], num_classes=c_dim)  # budama güvenliğini doğrula (bkz. Faz C2), out-of-range'de raise eder
    model = build_pruned_mapping_network(mapping_shard, num_classes=c_dim, embed_mode=circuit_config["embed_mode"])
    model.eval()

    # w'nin gerçek büyüklüğü — çağıran taraf (ör. scripts/replay_proofs.py)
    # max_abs_error'u buna oranlayıp bağıl hatayı raporluyor (Faz C3'teki
    # aynı desen, bkz. scripts/bench_circuit.py: max_abs_error_relative_pct).
    with torch.no_grad():
        w_abs_max = model(z, c).abs().max().item()

    onnx_path = work_dir / f"mapping_round{round_id}_{site}.onnx"
    t0 = time.perf_counter()
    export_to_onnx(model, z, c, str(onnx_path), opset_version=13)
    onnx_export_seconds = time.perf_counter() - t0
    print(f"[round_runner]  onnx_export: {onnx_export_seconds:.2f}s")

    input_json_path = work_dir / "input.json"
    write_json_file(build_multi_input_json(z, c), str(input_json_path))

    setup_result = run_ezkl_pipeline(onnx_path, input_json_path, work_dir, circuit_config["scale"])
    prove_result = run_prove_and_verify(input_json_path, setup_result["paths"], work_dir)

    sol_path, _abi_path = generate_solidity_verifier(setup_result["paths"], work_dir)
    t0 = time.perf_counter()
    compiled_verifier = compile_verifier_solidity(sol_path)
    solc_compile_seconds = time.perf_counter() - t0
    if compiled_verifier["exceeds_eip170"]:
        raise RuntimeError(
            f"[round_runner] round={round_id} site={site}: verifier bytecode EIP-170'i AŞIYOR "
            f"({compiled_verifier['deployed_bytecode_size']} > {EIP170_MAX_DEPLOYED_BYTECODE_SIZE}) — "
            f"ispat gönderilemez. configs/circuit.yaml'ın challenge_size_k=1 kullandığından emin ol."
        )

    deploy_client = Web3Client(round_manager_client.rpc_url)
    verifier_address, deploy_gas = deploy_client.deploy_bytecode(compiled_verifier["bytecode"], private_key)

    # proof.json'un İÇERİĞİ Faz B/C boyunca hiç ayrıştırılmadı (sadece
    # varlığı kontrol edilip ezkl.verify'e dosya yolu olarak verildi).
    # Zincire göndermek için burada İLK KEZ ayrıştırılıyor. Gerçek Colab
    # koşumunda gözlenen biçim: `proof` bir INT LİSTESİ (hex string
    # DEĞİL), `instances` LITTLE-ENDIAN hex string'ler — `parse_proof_bytes`/
    # `parse_public_inputs` hiçbirini VARSAYMAZ (bkz. circuits/ezkl_utils.py).
    with open(prove_result["proof_path"], encoding="utf-8") as f:
        proof_json = json.load(f)
    proof_bytes = parse_proof_bytes(proof_json)
    public_inputs = parse_public_inputs(proof_json)

    verified, submit_gas = round_manager_client.submit_proof(round_id, verifier_address, proof_bytes, public_inputs, private_key)

    timings = {
        "onnx_export": onnx_export_seconds,
        **setup_result["timings"],
        **prove_result["timings"],
        "solc_compile": solc_compile_seconds,
    }
    total_seconds = sum(timings.values())
    print(f"[round_runner]  === round={round_id} site={site} adım-adım süre dökümü (toplam {total_seconds:.2f}s) ===")
    for step, seconds in timings.items():
        print(f"[round_runner]    {step:<20} {seconds:>8.2f}s")

    return {
        "verifier_address": verifier_address,
        "deploy_gas": deploy_gas,
        "submit_gas": submit_gas,
        "verified": verified,
        "timings": timings,
        "circuit_stats": setup_result["circuit_stats"],
        "requested_scale": setup_result["requested_scale"],
        "realized_scale": setup_result["realized_scale"],
        "pk_size_bytes": setup_result["pk_size_bytes"],
        "vk_size_bytes": setup_result["vk_size_bytes"],
        "proof_size_bytes": prove_result["proof_size_bytes"],
        "deployed_bytecode_size": compiled_verifier["deployed_bytecode_size"],
        "used_solc_strategy": compiled_verifier.get("used_strategy"),
        "w_abs_max": w_abs_max,
    }


def run_round(
    round_id: int,
    sites: list,
    *,
    raw_root: str,
    stylegan_xl_repo: str,
    prev_global_state: dict,
    reputations: dict,
    circuit_config: dict,
    schedule_config: dict,
    round_manager_client,
    private_key: str,
    work_dir: Path,
    mode: str = "sampled",
) -> dict:
    from fl.round_replay import load_site_update
    from fl.shard_utils import extract_prefixed_state_dict

    print(f"\n=== Round {round_id} ===")
    global_hash = compute_weight_commitment(prev_global_state)
    global_cid = upload_state_dict_to_ipfs(prev_global_state, work_dir, f"global_round{round_id}.pt")

    challenge_seed, start_gas = round_manager_client.start_round(round_id, global_cid, global_hash, private_key)
    print(f"[round_runner] startRound: gas={start_gas} challenge_seed={challenge_seed.hex()}")

    z, c = build_challenge_z_c(challenge_seed, k=circuit_config["challenge_size_k"])

    site_schedule = build_round_schedule(
        round_id, sites, challenge_seed=challenge_seed, reputations=reputations, config=schedule_config, mode=mode
    )
    print(f"[round_runner] Ispat takvimi: {site_schedule}")

    site_states = {}
    chain_approved = {}
    proof_reports = {}

    for site in sites:
        full_state = load_site_update(raw_root, stylegan_xl_repo, round_id, sites.index(site))
        site_states[site] = full_state

        weight_commitment = compute_weight_commitment(full_state)
        update_cid = upload_state_dict_to_ipfs(full_state, work_dir, f"update_round{round_id}_{site}.pt")
        update_gas = round_manager_client.submit_update(round_id, update_cid, weight_commitment, private_key)
        print(f"[round_runner] site={site} submitUpdate: gas={update_gas} cid={update_cid}")

        if site_schedule.get(site, False):
            mapping_shard = extract_prefixed_state_dict(full_state, MAPPING_PREFIXES)
            report = generate_and_submit_proof(
                round_id=round_id,
                site=site,
                mapping_shard=mapping_shard,
                z=z,
                c=c,
                circuit_config=circuit_config,
                work_dir=work_dir,
                round_manager_client=round_manager_client,
                private_key=private_key,
            )
            proof_reports[site] = report
            print(f"[round_runner] site={site} submitProof: verified={report['verified']} gas={report['submit_gas']}")
            chain_approved[site] = report["verified"]
        else:
            print(f"[round_runner] site={site}: bu round'da ispat İSTENMEDİ (takvim).")
            chain_approved[site] = round_manager_client.is_site_eligible(site)

    aggregation = aggregate_round(
        prev_global_state, site_states, tau_norm_threshold=schedule_config["tau_norm_threshold"], chain_approved=chain_approved
    )

    aggregate_cid = upload_state_dict_to_ipfs(aggregation["global_state"], work_dir, f"aggregate_round{round_id}.pt")
    included_sites = aggregation["included_sites"]
    finalize_gas = round_manager_client.finalize_round(round_id, aggregate_cid, included_sites, private_key)
    print(f"[round_runner] finalizeRound: gas={finalize_gas} included={included_sites}")

    return {
        "round_id": round_id,
        "challenge_seed": challenge_seed.hex(),
        "site_schedule": site_schedule,
        "proof_reports": proof_reports,
        "aggregation": {k: v for k, v in aggregation.items() if k != "global_state"},
        "aggregate_cid": aggregate_cid,
        "new_global_state": aggregation["global_state"],
        "gas": {"start_round": start_gas, "finalize_round": finalize_gas},
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz D: bir round aralığını (replay) protokol üzerinden çalıştır.")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--schedule-config", default="configs/schedule.yaml")
    parser.add_argument("--circuit-config", default=DEFAULT_CIRCUIT_CONFIG)
    parser.add_argument("--deployment", default=None, help="varsayılan: {zk_root}/chain/deployment.json (scripts.deploy_contracts çıktısı)")
    parser.add_argument("--sites", required=True, help="virgülle ayrılmış zincir adresi listesi, site sırası = Faz A site index'i")
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--end-round", type=int, default=14)
    parser.add_argument("--mode", default="sampled", choices=("sampled", "full"))
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--progress-file", default=None, help="varsayılan: {zk_root}/chain/round_runner_progress.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from chain.client import RoundManagerClient
    from configs.loader import load_paths
    from fl.round_replay import load_site_update

    paths = load_paths(env=args.env, config_path=args.paths_config)
    schedule_config = load_schedule_config(args.schedule_config)
    circuit_config = load_circuit_config(args.circuit_config)

    deployment_path = args.deployment or os.path.join(paths["zk_root"], "chain", "deployment.json")
    with open(deployment_path, encoding="utf-8") as f:
        deployment = json.load(f)

    round_manager_client = RoundManagerClient(deployment["rpc_url"], deployment["round_manager_address"], deployment["round_manager_abi"])

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    work_dir = Path(args.work_dir or os.path.join(default_work_root(), "round_runner"))
    work_dir.mkdir(parents=True, exist_ok=True)

    progress_path = args.progress_file or os.path.join(paths["zk_root"], "chain", "round_runner_progress.json")
    progress = load_progress(progress_path)

    reputations = {site: schedule_config["reputation_initial"] for site in sites}
    prev_global_state = load_site_update(paths["raw_root"], paths["stylegan_xl_repo"], args.start_round, 0)

    for round_id in range(args.start_round, args.end_round + 1):
        key = str(round_id)
        if key in progress["rounds"]:
            print(f"[round_runner] round={round_id} zaten tamamlanmış, atlanıyor (progress.json).")
            continue

        result = run_round(
            round_id,
            sites,
            raw_root=paths["raw_root"],
            stylegan_xl_repo=paths["stylegan_xl_repo"],
            prev_global_state=prev_global_state,
            reputations=reputations,
            circuit_config=circuit_config,
            schedule_config=schedule_config,
            round_manager_client=round_manager_client,
            private_key=args.private_key,
            work_dir=work_dir,
            mode=args.mode,
        )

        for site, report in result["proof_reports"].items():
            reputations[site] = round_manager_client.get_reputation(site)

        prev_global_state = result["new_global_state"]
        progress["rounds"][key] = {k: v for k, v in result.items() if k != "new_global_state"}
        save_progress(progress, progress_path)
        print(f"[round_runner] round={round_id} tamamlandı, progress.json güncellendi: {progress_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
