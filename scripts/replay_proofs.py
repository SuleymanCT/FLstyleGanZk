#!/usr/bin/env python
"""Faz E: mevcut 15 round'un 60 shard'ı üzerinde, HİÇ EĞİTİM YAPMADAN,
ispat takvimini koşturup "full" (her round × her site) ile "staged"
(`orchestrator/schedule.py`'nin takvimi) modun maliyetini karşılaştırır.

CLAUDE.md madde 2/3 gereği bu script BİR EĞİTİM TURU BAŞLATMAZ —
`circuits.export_mapping.load_shard` ile Faz A'da ZATEN çıkarılmış
shard dosyalarını (`{shards_dir}/round_N/site_M.pt`) doğrudan okur.
Operasyonel devre parametreleri `configs/circuit.yaml`'dan (k=1,
matmul, scale=8 — Faz C3'ün kararı), itibar/takvim parametreleri
`configs/schedule.yaml`'dan gelir.

## Mimari: bench_circuit.py'nin (Faz C3) alt-süreç deseni + round_runner.py'nin (Faz D) ispat mantığı

Her (round,site) ispatı KENDİ alt sürecinde çalışır — `scripts/bench_circuit.py`'de
kanıtlanmış gerekçelerle AYNI: (1) tepe RAM kombinasyon başına temiz
ölçülebiliyor, (2) ezkl'nin Rust tarafının bastığı loglar (decomposition
uyarıları, `max_abs_error`) alt sürecin GERÇEK stdout'undan güvenilir
yakalanıyor, (3) bir ispat çökerse/panikler se sadece o alt süreç ölür,
diğerlerine devam edilir. Alt sürecin GERÇEK ispat/deploy/submit
mantığı `orchestrator.round_runner.generate_and_submit_proof`'u (Faz
D'de `tests/test_contracts.py` ile doğrulanmış) DOĞRUDAN çağırır —
hiçbir ezkl/solc/chain adımı burada YENİDEN YAZILMAZ.

**anvil TEK bir süreçte, TÜM koşum boyunca canlı kalır** (her ispat için
yeniden başlatılmaz) — `RoundManager` bir kez deploy edilir, her round
için `startRound` GERÇEKTEN çağrılıp GERÇEK bir `challengeSeed` alınır
(alt süreçlere hex string olarak aktarılır, `orchestrator.challenge.build_challenge_z_c`
ile deterministik (z,c)'ye çevrilir — aynı seed HER ZAMAN aynı (z,c)
verir).

**"staged" modda hangi site'ların ispat üreteceği ÖNCEDEN
hesaplanamaz** — `orchestrator.schedule.must_prove`'un rastgele bileşeni
GERÇEK `challengeSeed`'e bağlı, o da GERÇEK `startRound` çağrısı
YAPILMADAN bilinemez (`blockhash(block.number-1)` türevi). Bu yüzden
takvim kararı HER ROUND İÇİN, o round'un GERÇEK `challengeSeed`'i
alındıktan HEMEN SONRA, çalışma zamanında veriliyor.

## Disk yönetimi (Faz C3 emsaliyle)

Varsayılan çalışma kökü `/content` (Colab'ın yerel diski, Drive DEĞİL) —
büyük ara dosyalar (pk, derlenmiş devre, witness) `--keep-artifacts`
yoksa her ispattan sonra silinir. Sadece küçük sonuç dosyaları
(`{key}.json`, `.log.txt`, `replay_results_<mod>.json`) Drive'daki
`{zk_root}/replay`'e yazılır.

BİLİNÇLİ TEST SINIRI: gerçek shard dosyaları (Faz A çıktısı), ezkl,
anvil, solc, web3 gerektirir — SADECE Colab'da çalışır/test edilir. Saf
yardımcılar (`parse_int_range`, `replay_combo_key`, `compute_mode_totals`,
`render_mode_comparison_table`, `render_staged_distribution`)
`tests/test_replay_proofs.py`'de yerelde GERÇEKTEN test edilir.

Kullanım (Colab'da):
    python -m scripts.replay_proofs --env colab --mode full --only-first
    python -m scripts.replay_proofs --env colab --mode full
    python -m scripts.replay_proofs --env colab --mode staged
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from configs.loader import load_paths
from orchestrator.schedule import load_schedule_config
from scripts.bench_circuit import (
    classify_exception_status,
    cleanup_work_dir,
    count_decomposition_warnings,
    default_work_root,
    extract_max_abs_error,
    load_bench_results as load_replay_results,
    measure_peak_rss_kb,
    save_bench_results as save_replay_results,
    write_json_file,
)
from storage.pathguard import assert_writable

DEFAULT_ROUNDS = "0-14"
DEFAULT_SITES = "0-3"
DEFAULT_TIMEOUT_SECONDS = 1800.0
EZKL_TIMING_KEYS = ("gen_settings", "calibrate_settings", "compile_circuit", "setup", "gen_witness", "prove", "verify_offchain")


# ---------------------------------------------------------------------------
# Saf/testable yardımcılar (ezkl/anvil/solc/web3 GEREKTİRMEZ)
# ---------------------------------------------------------------------------


def parse_int_range(raw: str) -> list[int]:
    """`"0-14"` -> `[0,1,...,14]`, `"1,3,5"` -> `[1,3,5]`, ikisi
    birlikte de olabilir (`"0-2,5"`)."""
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            if end < start:
                raise ValueError(f"Geçersiz aralık: {part!r} (bitiş < başlangıç)")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError(f"Boş/geçersiz aralık ifadesi: {raw!r}")
    return values


def replay_combo_key(round_id: int, site_index: int) -> str:
    return f"round{round_id}_site{site_index}"


def compute_mode_totals(results: dict) -> dict:
    """Bir modun (full/staged) TÜM ispat sonuçlarını özetler — makalenin
    ana karşılaştırma tablosunun ham girdisi."""
    num_success = sum(1 for r in results.values() if r.get("status") == "success")
    total_ezkl_seconds = 0.0
    total_solc_seconds = 0.0
    total_deploy_gas = 0
    total_verify_gas = 0
    for r in results.values():
        timings = r.get("timings") or {}
        total_ezkl_seconds += sum(timings.get(k) or 0.0 for k in EZKL_TIMING_KEYS)
        total_solc_seconds += timings.get("solc_compile") or 0.0
        total_deploy_gas += r.get("deploy_gas") or 0
        total_verify_gas += r.get("submit_gas") or 0
    return {
        "num_proofs": len(results),
        "num_success": num_success,
        "total_ezkl_seconds": total_ezkl_seconds,
        "total_solc_seconds": total_solc_seconds,
        "total_deploy_gas": total_deploy_gas,
        "total_verify_gas": total_verify_gas,
        "total_chain_cost_gas": total_deploy_gas + total_verify_gas,
    }


def render_mode_comparison_table(totals_by_mode: dict) -> str:
    """Makalenin ana tablosu: `mod | ispat sayısı | toplam ezkl süresi |
    toplam solc süresi | toplam deploy gas | toplam verify gas | toplam
    zincir maliyeti | tasarruf %`. "full" modu referans alınır — diğer
    modların tasarruf yüzdesi ona göre hesaplanır. "full" sonucu HENÜZ
    yoksa (`totals_by_mode`'da), tasarruf sütunu "-" kalır (uydurulmaz)."""
    headers = [
        "mod", "ispat sayısı", "başarılı", "toplam ezkl süresi (s)", "toplam solc süresi (s)",
        "toplam deploy gas", "toplam verify gas", "toplam zincir maliyeti (gas)", "tasarruf %",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    full_cost = totals_by_mode.get("full", {}).get("total_chain_cost_gas")

    for mode, totals in totals_by_mode.items():
        cost = totals["total_chain_cost_gas"]
        if mode == "full" or not full_cost:
            savings = "-"
        else:
            savings = f"{100.0 * (1 - cost / full_cost):.1f}"
        row = [
            mode, str(totals["num_proofs"]), str(totals["num_success"]),
            f"{totals['total_ezkl_seconds']:.1f}", f"{totals['total_solc_seconds']:.1f}",
            str(totals["total_deploy_gas"]), str(totals["total_verify_gas"]), str(cost), savings,
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def render_staged_distribution(results: dict) -> str:
    """"staged" modda hangi round'da hangi site'ların GERÇEKTEN ispat
    ürettiğini gösteren tablo — takvimin fiilen nasıl dağıldığını
    görmek için."""
    by_round: dict = {}
    for r in results.values():
        combo = r.get("combo") or {}
        round_id, site_index = combo.get("round_id"), combo.get("site_index")
        if round_id is None or site_index is None:
            continue
        by_round.setdefault(round_id, []).append(site_index)

    lines = ["| round | ispat üreten site'lar |", "|---|---|"]
    for round_id in sorted(by_round):
        sites_str = ", ".join(f"site_{s}" for s in sorted(by_round[round_id]))
        lines.append(f"| {round_id} | {sites_str} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ezkl/anvil/solc/web3'e bağımlı adımlar (BİLİNÇLİ TEST SINIRI — sadece Colab)
# ---------------------------------------------------------------------------


def run_worker(args: argparse.Namespace) -> int:
    """TEK bir (round,site) ispatını uçtan uca çalıştırır. Ebeveyn süreç
    bunu `--worker` bayrağıyla bir alt süreç olarak başlatır (bkz. modül
    docstring'i — izolasyon gerekçeleri)."""
    from chain.client import RoundManagerClient
    from circuits.export_mapping import load_shard
    from orchestrator.challenge import build_challenge_z_c
    from orchestrator.round_runner import generate_and_submit_proof, load_circuit_config

    combo = {"round_id": args.round_id, "site_index": args.site_index}
    key = replay_combo_key(args.round_id, args.site_index)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"combo": combo, "status": "failed", "error_summary": None}
    t_start = time.perf_counter()

    try:
        circuit_config = load_circuit_config(args.circuit_config)
        shard = load_shard(args.shards_dir, args.round_id, args.site_index)

        challenge_seed = bytes.fromhex(args.challenge_seed_hex)
        z, c = build_challenge_z_c(challenge_seed, k=circuit_config["challenge_size_k"])
        del challenge_seed
        gc.collect()

        with open(args.round_manager_abi_file, encoding="utf-8") as f:
            abi = json.load(f)
        round_manager_client = RoundManagerClient(args.rpc_url, args.round_manager_address, abi)

        report = generate_and_submit_proof(
            round_id=args.round_id,
            site=f"site_{args.site_index}",
            mapping_shard=shard,
            z=z,
            c=c,
            circuit_config=circuit_config,
            work_dir=work_dir,
            round_manager_client=round_manager_client,
            private_key=args.private_key,
        )
        result.update(report)
        result["status"] = "success"

    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa: BLE001 - pyo3 PanicException DAHIL tüm hatalar burada yakalanıp raporlanıyor, koşu düşmüyor
        result["status"] = classify_exception_status(e)
        result["error_summary"] = f"{type(e).__module__}.{type(e).__name__}: {e}"
        print(f"[replay_proofs] KOMBİNASYON {result['status'].upper()} ({key}): {result['error_summary']}")
        print(traceback.format_exc())

    finally:
        result["total_wall_seconds"] = time.perf_counter() - t_start
        result["peak_rss_kb"] = measure_peak_rss_kb()
        result.update(cleanup_work_dir(work_dir, keep_artifacts=args.keep_artifacts))
        write_json_file(result, args.result_out)
        print(f"[replay_proofs] Sonuç yazıldı: {args.result_out}")
        gc.collect()

    return 0 if result["status"] == "success" else 1


def run_combo_in_subprocess(
    round_id: int,
    site_index: int,
    challenge_seed: bytes,
    *,
    rpc_url: str,
    round_manager_address: str,
    abi_file: Path,
    private_key: str,
    shards_dir: str,
    circuit_config_path: str,
    work_root: Path,
    replay_dir: Path,
    timeout: float,
    keep_artifacts: bool,
) -> dict:
    key = replay_combo_key(round_id, site_index)
    combo_work_dir = work_root / key
    combo_work_dir.mkdir(parents=True, exist_ok=True)
    results_root = replay_dir / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    result_path = results_root / f"{key}.json"
    log_path = results_root / f"{key}.log.txt"

    cmd = [
        sys.executable, "-m", "scripts.replay_proofs", "--worker",
        "--round-id", str(round_id), "--site-index", str(site_index),
        "--challenge-seed-hex", challenge_seed.hex(),
        "--rpc-url", rpc_url, "--round-manager-address", round_manager_address,
        "--round-manager-abi-file", str(abi_file), "--private-key", private_key,
        "--shards-dir", shards_dir, "--circuit-config", circuit_config_path,
        "--work-dir", str(combo_work_dir), "--result-out", str(result_path),
    ]
    if keep_artifacts:
        cmd.append("--keep-artifacts")

    print(f"\n=== Kombinasyon: {key} (timeout={timeout}s) ===")
    print(f"[replay_proofs] Komut: {' '.join(cmd)}")

    popen_kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True  # anvil ile ilgisiz - sadece bu alt sürecin kendi grubu icin
    proc = subprocess.Popen(cmd, **popen_kwargs)

    timed_out = False
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(f"[replay_proofs] '{key}' {timeout}s içinde bitmedi, sonlandırılıyor.")
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        stdout, _ = proc.communicate()

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(stdout or "")

    if result_path.is_file():
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = {
            "combo": {"round_id": round_id, "site_index": site_index},
            "status": "failed",
            "error_summary": "Worker süreci sonuç dosyası yazmadan sonlandı (çökme/timeout-kill).",
            "timings": {},
            "peak_rss_kb": None,
        }

    if timed_out:
        result["status"] = "failed"
        result["error_summary"] = f"{result.get('error_summary') or ''} [TIMEOUT: {timeout}s aşıldı]".strip()

    result["returncode"] = proc.returncode
    result["decomposition_warning_count"] = count_decomposition_warnings(stdout or "")
    result["log_path"] = str(log_path)

    max_abs_error = extract_max_abs_error(stdout or "")
    if max_abs_error is None and re.search(r"max_abs_error", stdout or "", re.IGNORECASE):
        print(f"[replay_proofs] UYARI: '{key}' log'unda 'max_abs_error' geçiyor ama ayrıştırılamadı.")
        result["max_abs_error"] = "yakalanamadi"
        result["max_abs_error_relative_pct"] = "yakalanamadi"
    else:
        result["max_abs_error"] = max_abs_error
        w_abs_max = result.get("w_abs_max")
        if max_abs_error is not None and w_abs_max:
            result["max_abs_error_relative_pct"] = 100.0 * max_abs_error / w_abs_max
        else:
            result["max_abs_error_relative_pct"] = None

    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz E: mevcut round shard'ları üzerinde ispat takvimini (full/staged) koştur.")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--schedule-config", default="configs/schedule.yaml")
    parser.add_argument("--circuit-config", default="configs/circuit.yaml")
    parser.add_argument("--round-manager-sol", default="contracts/RoundManager.sol")
    parser.add_argument("--shards-dir", default=None, help="varsayılan: {shards_dir} (configs/paths.yaml)")
    parser.add_argument("--replay-dir", default=None, help="varsayılan: {zk_root}/replay (Drive) — SADECE küçük sonuç dosyaları")
    parser.add_argument("--work-root", default=None, help="varsayılan: /content/zk_bench_work (yerel disk) — BÜYÜK ara dosyalar")
    parser.add_argument("--mode", required=True, choices=["full", "staged"])
    parser.add_argument("--rounds", default=DEFAULT_ROUNDS)
    parser.add_argument("--sites", default=DEFAULT_SITES)
    parser.add_argument("--only-first", action="store_true", help="sadece ilk kombinasyonu koş")
    parser.add_argument("--limit", type=int, default=None, help="en fazla N ispat koştur (kalanını atla)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="ispat başına saniye")
    parser.add_argument("--force", action="store_true", help="zaten sonuçlandırılmış kombinasyonları yeniden çalıştır")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--markdown-out", default="docs/phase_e_replay.md")
    # Aşağıdakiler SADECE alt süreç (--worker) tarafından kullanılır.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--round-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--site-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--challenge-seed-hex", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rpc-url", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--round-manager-address", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--round-manager-abi-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--private-key", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-out", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.worker:
        return run_worker(args)

    from chain.anvil import AnvilProcess
    from chain.client import RoundManagerClient, Web3Client
    from chain.solc import compile_with_fallback_strategies
    from circuits.export_mapping import load_shard
    from orchestrator.round_runner import compute_weight_commitment, load_circuit_config
    from orchestrator.schedule import build_round_schedule

    paths = load_paths(env=args.env, config_path=args.paths_config)
    schedule_config = load_schedule_config(args.schedule_config)
    circuit_config = load_circuit_config(args.circuit_config)

    shards_dir = args.shards_dir or paths["shards_dir"]
    replay_dir = Path(args.replay_dir or os.path.join(paths["zk_root"], "replay"))
    work_root = Path(args.work_root or default_work_root())

    rounds = parse_int_range(args.rounds)
    sites = parse_int_range(args.sites)
    if args.only_first:
        rounds, sites = rounds[:1], sites[:1]

    print(f"[replay_proofs] mod={args.mode} rounds={rounds} sites={sites}")
    print(f"[replay_proofs] shards_dir={shards_dir}")
    print(f"[replay_proofs] replay_dir={replay_dir} (KÜÇÜK sonuç dosyaları — Drive)")
    print(f"[replay_proofs] work_root={work_root} (BÜYÜK ara dosyalar — yerel disk)")
    print(f"[replay_proofs] timeout={args.timeout}s limit={args.limit} force={args.force} keep_artifacts={args.keep_artifacts}")

    assert_writable(str(replay_dir))
    replay_dir.mkdir(parents=True, exist_ok=True)
    assert_writable(str(work_root))
    work_root.mkdir(parents=True, exist_ok=True)

    results_path = str(replay_dir / f"replay_results_{args.mode}.json")
    all_results = load_replay_results(results_path)

    reputations = {s: schedule_config["reputation_initial"] for s in sites}
    processed = 0

    print("[replay_proofs] anvil başlatılıyor (TEK süreç, tüm koşum boyunca canlı kalacak)...")
    with AnvilProcess() as anvil:
        if len(anvil.accounts) < len(sites) + 1:
            raise RuntimeError(f"anvil yeterli hesap üretmedi: {len(anvil.accounts)} hesap, {len(sites) + 1} gerekli.")
        owner_address, owner_key = anvil.accounts[0]
        site_accounts = {s: anvil.accounts[i + 1] for i, s in enumerate(sites)}

        deploy_client = Web3Client(anvil.rpc_url)
        print(f"[replay_proofs] '{args.round_manager_sol}' derleniyor...")
        compiled_rm = compile_with_fallback_strategies(Path(args.round_manager_sol))
        proof_schedule = schedule_config["proof_schedule"]
        round_manager_address, deploy_rm_gas = deploy_client.deploy_contract(
            compiled_rm["abi"],
            compiled_rm["bytecode"],
            (schedule_config["reputation_initial"], schedule_config["reputation_penalty"], schedule_config["reputation_bonus"], proof_schedule["reputation_threshold"]),
            owner_key,
        )
        print(f"[replay_proofs] RoundManager deploy edildi: {round_manager_address} (gas={deploy_rm_gas})")

        round_manager_client = RoundManagerClient(anvil.rpc_url, round_manager_address, compiled_rm["abi"])
        for s in sites:
            round_manager_client.register_site(site_accounts[s][0], owner_key)

        abi_file = work_root / "round_manager_abi.json"
        write_json_file(compiled_rm["abi"], str(abi_file))

        for round_id in rounds:
            if args.limit is not None and processed >= args.limit:
                print(f"[replay_proofs] --limit {args.limit} doldu, durduruluyor.")
                break

            challenge_seed, start_gas = round_manager_client.start_round(
                round_id, f"replay-round-{round_id}", (round_id).to_bytes(32, "big"), owner_key
            )
            print(f"[replay_proofs] round={round_id} startRound: gas={start_gas} challenge_seed={challenge_seed.hex()}")

            if args.mode == "full":
                sites_to_prove = list(sites)
            else:
                schedule = build_round_schedule(
                    round_id, sites, challenge_seed=challenge_seed, reputations=reputations, config=schedule_config, mode="sampled"
                )
                sites_to_prove = [s for s in sites if schedule[s]]
                print(f"[replay_proofs] round={round_id} ispat takvimi: {schedule}")

            for site_index in sites_to_prove:
                if args.limit is not None and processed >= args.limit:
                    break

                key = replay_combo_key(round_id, site_index)
                if key in all_results and not args.force:
                    print(f"[replay_proofs] '{key}' zaten sonuçlandırılmış, atlanıyor (--force ile yeniden çalıştır).")
                    continue

                site_address, site_key = site_accounts[site_index]

                try:
                    shard = load_shard(shards_dir, round_id, site_index)
                    weight_commitment = compute_weight_commitment(shard)
                    del shard
                    gc.collect()
                    round_manager_client.submit_update(round_id, f"replay-update-{round_id}-{site_index}", weight_commitment, site_key)
                except Exception as e:  # noqa: BLE001 - bu (round,site) başarısız işaretlenip devam ediliyor
                    result = {
                        "combo": {"round_id": round_id, "site_index": site_index},
                        "status": "failed",
                        "error_summary": f"submitUpdate/shard öncesi hazırlık başarısız: {type(e).__name__}: {e}",
                    }
                    print(f"[replay_proofs] '{key}' BAŞARISIZ (submitUpdate öncesi): {result['error_summary']}")
                else:
                    result = run_combo_in_subprocess(
                        round_id, site_index, challenge_seed,
                        rpc_url=anvil.rpc_url, round_manager_address=round_manager_address, abi_file=abi_file,
                        private_key=site_key, shards_dir=shards_dir, circuit_config_path=args.circuit_config,
                        work_root=work_root, replay_dir=replay_dir, timeout=args.timeout, keep_artifacts=args.keep_artifacts,
                    )
                    if result.get("verified"):
                        reputations[site_index] = round_manager_client.get_reputation(site_address)

                all_results[key] = result
                save_replay_results(all_results, results_path)
                processed += 1
                print(
                    f"[replay_proofs] '{key}' bitti: durum={result['status']} "
                    f"tepe_RAM_KB={result.get('peak_rss_kb')} verified={result.get('verified')}"
                )
                gc.collect()

    print(f"\n=== Faz E ({args.mode}) sonuçları ===")
    totals = compute_mode_totals(all_results)
    print(json.dumps(totals, indent=2, ensure_ascii=False))

    other_mode = "staged" if args.mode == "full" else "full"
    other_results_path = replay_dir / f"replay_results_{other_mode}.json"
    totals_by_mode = {args.mode: totals}
    if other_results_path.is_file():
        other_results = load_replay_results(str(other_results_path))
        totals_by_mode[other_mode] = compute_mode_totals(other_results)
    else:
        print(f"[replay_proofs] '{other_mode}' modu henüz koşulmadı — karşılaştırma tablosu bu mod için EKSİK kalacak.")

    ordered = {m: totals_by_mode[m] for m in ("full", "staged") if m in totals_by_mode}
    comparison_table = render_mode_comparison_table(ordered)
    print("\n" + comparison_table)

    md_parts = [f"# Faz E — İspat Takvimi Maliyet Karşılaştırması\n\n", "## Mod karşılaştırma tablosu\n\n", comparison_table]
    if "staged" in totals_by_mode:
        staged_results = all_results if args.mode == "staged" else load_replay_results(str(replay_dir / "replay_results_staged.json"))
        md_parts += ["\n## \"staged\" modda round başına ispat dağılımı\n\n", render_staged_distribution(staged_results)]

    md_path = args.markdown_out
    assert_writable(md_path)
    os.makedirs(os.path.dirname(os.path.abspath(md_path)) or ".", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_parts))
    print(f"[replay_proofs] Yazıldı: {md_path}")
    print(f"[replay_proofs] Yazıldı: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
