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
# NOT: "get_srs" ve "onnx_export" eskiden bu listede YOKTU — total_ezkl_seconds
# gerçekte ölçülen sürenin BİR KISMINI atlıyordu (Faz E'nin ilk tam-ispat
# koşumunda 467.5s toplam gözlendi, Faz C3'ün ~75s'lik referansının 6 katı
# — bkz. docs/phase_e_timing_investigation.md). Adım-adım dökümün TAMAMI
# artık burada.
EZKL_TIMING_KEYS = (
    "onnx_export", "gen_settings", "calibrate_settings", "compile_circuit",
    "get_srs", "setup", "gen_witness", "prove", "verify_offchain",
)

# scripts/replay_proofs.py'ye ÖZGÜ anvil ömür-döngüsü varsayılanları —
# configs/schedule.yaml: replay_infra ile override edilebilir (bkz.
# get_replay_infra_config). Faz E'nin gerçek Colab koşumunda anvil 36
# ispat sonrası yanıt vermez oldu (RPC ReadTimeout) — bkz.
# docs/phase_e_replay.md "Altyapı kısıtları" bölümü.
DEFAULT_ANVIL_RESTART_INTERVAL = 10
DEFAULT_RPC_TIMEOUT_SECONDS = 120.0
DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0


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


# Faz E'nin ilk tam mod koşumunda anvil ÇÖKMESİ yaşandı (36 ispat sonrası
# RPC yanıt vermez oldu, bkz. docs/phase_e_infra_notes.md) — ham veri
# incelemesi bunun SADECE RPC'yi değil, AYNI makinede koşan ezkl'i de
# (bellek baskısı/swap) yavaşlattığını ortaya çıkardı: setup+prove ~74s'den
# ~260s'ye çıkmış (bkz. docs/phase_e_report.md, "Sistematik teşhis"
# bölümü). `run_id` BUNDAN SONRAKİ koşumlar için `main()`'in başında
# üretilip her sonuca yazılıyor — ama ESKİ kayıtlarda (bu koşumdan önce
# üretilmiş) bu alan YOK. Bu yüzden geriye dönük sınıflandırma HER ZAMAN
# `timings` üzerinden yapılır, `run_id`'nin varlığına bağlı DEĞİL.
ENVIRONMENT_DEGRADED_THRESHOLD_SECONDS = 150.0


def classify_environment_health(timings: dict) -> str:
    """`setup+prove` toplamı `ENVIRONMENT_DEGRADED_THRESHOLD_SECONDS`'ı
    aşıyorsa `"degraded"` (anvil'in bellek şişmesinin ezkl'i de
    yavaşlattığı, GERÇEK ortam — uydurma bir etiket değil, ölçülen
    veriden ayrıştırılıyor), aşmıyorsa `"healthy"`. `setup` VE `prove`
    ikisi de yoksa (başarısız/eksik kayıt) `"unknown"` — sessizce
    `"healthy"` SAYILMAZ, sınıflandırma için yeterli veri olmadığı
    açıkça işaretlenir."""
    setup = timings.get("setup")
    prove = timings.get("prove")
    if setup is None and prove is None:
        return "unknown"
    total = (setup or 0.0) + (prove or 0.0)
    return "degraded" if total > ENVIRONMENT_DEGRADED_THRESHOLD_SECONDS else "healthy"


def split_by_environment_health(results: dict) -> dict:
    """Sonuçları `classify_environment_health`'e göre üçe ayırır
    (`"healthy"`/`"degraded"`/`"unknown"`) — normalize tablo SADECE
    `"healthy"` kısmı kullanır."""
    buckets: dict = {"healthy": {}, "degraded": {}, "unknown": {}}
    for key, r in results.items():
        health = classify_environment_health(r.get("timings") or {})
        buckets[health][key] = r
    return buckets


def compute_healthy_avg_ezkl_seconds(results: dict) -> float | None:
    """SADECE başarılı VE `"healthy"` ortamda ölçülmüş ispatların ezkl
    süresi ortalaması — normalize tablonun "ispat başına sabit süre"
    varsayımının GERÇEK, ÖLÇÜLMÜŞ değeri (74s'yi VARSAYMAZ, mevcut
    healthy verilerden HESAPLAR). Hiç healthy+success kayıt yoksa
    `None` döner — çağıran taraf bunu "normalize edilemiyor" olarak
    ele almalı, 74 gibi sabit bir sayı UYDURMAMALI."""
    healthy = split_by_environment_health(results)["healthy"]
    successful = [r for r in healthy.values() if r.get("status") == "success"]
    if not successful:
        return None
    per_proof = [sum((r.get("timings") or {}).get(k) or 0.0 for k in EZKL_TIMING_KEYS) for r in successful]
    return sum(per_proof) / len(per_proof)


def render_normalized_comparison_table(totals_by_mode: dict, healthy_avg_ezkl_seconds: float) -> str:
    """Ham toplamlar tablosundaki `total_ezkl_seconds`'ı, BOZUK ortamda
    (anvil çökmesi sırasında) ölçülmüş ispatların şişirdiği toplam
    yerine, `ispat_sayısı * healthy_avg_ezkl_seconds` ile YENİDEN
    hesaplayıp gösterir. Gas sütunları DEĞİŞMEZ — deploy/verify gas'ı
    makine bellek durumundan ETKİLENMEZ (sabit devre/verifier boyutuna
    bağlı), bu yüzden normalizasyon SADECE süre tarafını düzeltir. İki
    ayrı tasarruf sütunu (süre/gas) BİLEREK ayrı tutulur — normalize
    edildikten sonra ikisinin YAKINSADIĞINI (ikisi de ~ispat sayısı
    oranına eşit) GÖSTERMEK bu tablonun asıl amacı."""
    headers = [
        "mod", "ispat sayısı", "normalize ezkl süresi (s)", "toplam zincir maliyeti (gas)",
        "süre tasarrufu % (normalize)", "gas tasarrufu %",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    full_totals = totals_by_mode.get("full")
    full_time = full_totals["num_proofs"] * healthy_avg_ezkl_seconds if full_totals else None
    full_cost = full_totals.get("total_chain_cost_gas") if full_totals else None

    for mode, totals in totals_by_mode.items():
        normalized_ezkl_seconds = totals["num_proofs"] * healthy_avg_ezkl_seconds
        cost = totals["total_chain_cost_gas"]
        if mode == "full" or not full_time:
            time_savings = "-"
        else:
            time_savings = f"{100.0 * (1 - normalized_ezkl_seconds / full_time):.1f}"
        if mode == "full" or not full_cost:
            gas_savings = "-"
        else:
            gas_savings = f"{100.0 * (1 - cost / full_cost):.1f}"
        row = [
            mode, str(totals["num_proofs"]), f"{normalized_ezkl_seconds:.1f}", str(cost),
            time_savings, gas_savings,
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


def render_failure_summary(results: dict) -> str:
    """Başarısız/çöken TÜM kombinasyonları hata özetiyle birlikte listeler
    — Faz E'nin ilk koşumunda "hata mesajı hiç basılmadı" sorununun
    kapanış raporunda da tekrarlanmaması için (konsola VE markdown'a
    yazılır)."""
    failures = {key: r for key, r in results.items() if r.get("status") != "success"}
    if not failures:
        return "Tüm kombinasyonlar başarılı — başarısızlık yok.\n"

    lines = ["| kombinasyon | durum | hata özeti |", "|---|---|---|"]
    for key, r in failures.items():
        error = r.get("error_summary") or r.get("error") or "-"
        error = str(error).replace("|", "\\|").replace("\n", " ")[:300]
        lines.append(f"| {key} | {r.get('status', '?')} | {error} |")
    return "\n".join(lines) + "\n"


def render_infra_events(events: list) -> str:
    """Anvil'in yeniden başlatıldığı/sağlıksız bulunduğu HER olayı
    listeler — "kesinti/yeniden başlatma sayısı" raporda bir "altyapı
    kısıtı" olarak AÇIKÇA görünsün diye (kullanıcının açık isteği,
    gizlenmez)."""
    if not events:
        return "Hiç yeniden başlatma/sağlık kontrolü olayı olmadı — anvil tüm koşum boyunca TEK bir segmentte kaldı.\n"

    lines = [f"Toplam {len(events)} altyapı olayı (yeniden başlatma/sağlık kontrolü):", "", "| segment | tür | bağlam |", "|---|---|---|"]
    for event in events:
        event = dict(event)
        segment_index = event.pop("segment_index", "?")
        event_type = event.pop("type", "?")
        context = ", ".join(f"{k}={v}" for k, v in event.items())
        lines.append(f"| {segment_index} | {event_type} | {context} |")
    return "\n".join(lines) + "\n"


def get_replay_infra_config(schedule_config: dict) -> dict:
    """`configs/schedule.yaml: replay_infra` bölümünü okur — YOKSA (eski
    bir schedule.yaml, ya da sadece round_runner.py'nin kullandığı bir
    kopya) sessizce varsayılanlara düşer, hata VERMEZ (bu bölüm SADECE
    replay_proofs.py'ye özgü, protokolün gerektirdiği bir alan değil)."""
    infra = schedule_config.get("replay_infra") or {}
    return {
        "anvil_restart_interval": infra.get("anvil_restart_interval", DEFAULT_ANVIL_RESTART_INTERVAL),
        "rpc_timeout_seconds": infra.get("rpc_timeout_seconds", DEFAULT_RPC_TIMEOUT_SECONDS),
        "health_check_timeout_seconds": infra.get("health_check_timeout_seconds", DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS),
    }


def should_restart_segment(proofs_in_segment: int, restart_interval: int) -> bool:
    """`restart_interval <= 0` -> asla planlı yeniden başlatma yapma
    (0/negatif = devre dışı bırakma anahtarı)."""
    return restart_interval > 0 and proofs_in_segment >= restart_interval


def round_fully_done(round_id: int, sites: list, all_results: dict, force: bool) -> bool:
    """Bir round'un TÜM site'ları için zaten sonuç varsa `startRound`
    çağrısını bile ATLAMAK üzere kullanılır (resume sırasında gereksiz
    gas/zaman harcamamak için) — "full" modda anlamlı bir kısayol;
    "staged" modda bazı round'larda hiç ispat gerekmeyebileceğinden
    (ör. tüm site'lar örneklemede elenmiş olabilir) bu fonksiyon
    genellikle `False` döner ve normal akış (kendi içinde zaten doğru
    şekilde ATLAMA yapan per-site kontrolü) devam eder — YANLIŞ bir
    atlamaya yol AÇMAZ, sadece bir hızlandırma fırsatını KAÇIRABİLİR."""
    if force:
        return False
    return all(replay_combo_key(round_id, s) in all_results for s in sites)


def check_rpc_alive(rpc_url: str, timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS) -> bool:
    """`eth_blockNumber` ile anvil'in GERÇEKTEN yanıt verip vermediğini
    KISA bir zaman aşımıyla kontrol eder — asıl RPC işlem zaman aşımından
    (`rpc_timeout_seconds`, dakikalar sürebilir) BİLEREK AYRI: amaç HIZLI
    bir "hayatta mı" sorgusu, uzun uzun beklemek DEĞİL. `requests`'i
    (web3.py'nin de kullandığı HTTP kütüphanesi) DOĞRUDAN kullanır —
    `Web3Client` kurmak (kendi `is_connected()` kontrolüyle) BİZİM
    `timeout`'umuzu değil web3.py'nin varsayılanını kullanabilirdi."""
    import requests

    try:
        response = requests.post(
            rpc_url,
            json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
            timeout=timeout,
        )
        response.raise_for_status()
        return "result" in response.json()
    except Exception:  # noqa: BLE001 - "canlı değil" sinyali, hangi hata olduğu önemli değil
        return False


# ---------------------------------------------------------------------------
# ezkl/anvil/solc/web3'e bağımlı adımlar (BİLİNÇLİ TEST SINIRI — sadece Colab)
# ---------------------------------------------------------------------------


def run_worker(args: argparse.Namespace) -> int:
    """TEK bir (round,site) ispatını uçtan uca çalıştırır. Ebeveyn süreç
    bunu `--worker` bayrağıyla bir alt süreç olarak başlatır (bkz. modül
    docstring'i — izolasyon gerekçeleri).

    EN DIŞ SEVİYEDE `try/except BaseException` kullanır (importlar DAHİL,
    `work_dir.mkdir()` DAHİL) — Faz E'nin ilk Colab koşumunda bir işçi
    süreç `result.json` HİÇ YAZMADAN, HİÇBİR HATA MESAJI OLMADAN
    başarısız oldu; kök sebep muhtemelen bu fonksiyonun eskiden `try`
    bloğunun DIŞINDA duran importları/`work_dir` kurulumuydu — oradaki
    bir hata hiçbir yerde yakalanmıyordu. Artık HİÇBİR KOD YOLU bu
    fonksiyonun try/except/finally'sinin dışında kalmıyor."""
    combo = {"round_id": args.round_id, "site_index": args.site_index}
    key = replay_combo_key(args.round_id, args.site_index)
    result: dict = {"combo": combo, "status": "failed", "error_summary": None, "error": None, "traceback": None}
    t_start = time.perf_counter()
    work_dir: Path | None = None

    try:
        from chain.client import RoundManagerClient
        from circuits.export_mapping import load_shard
        from orchestrator.challenge import build_challenge_z_c
        from orchestrator.round_runner import generate_and_submit_proof, load_circuit_config

        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

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
        error_text = f"{type(e).__module__}.{type(e).__name__}: {e}"
        result["error_summary"] = error_text
        result["error"] = error_text
        result["traceback"] = traceback.format_exc()
        print(f"[replay_proofs] KOMBİNASYON {result['status'].upper()} ({key}): {error_text}", flush=True)
        print(result["traceback"], flush=True)

    finally:
        result["total_wall_seconds"] = time.perf_counter() - t_start
        result["peak_rss_kb"] = measure_peak_rss_kb()
        if work_dir is not None:
            try:
                result.update(cleanup_work_dir(work_dir, keep_artifacts=args.keep_artifacts))
            except Exception as cleanup_error:  # noqa: BLE001 - temizlik başarısızlığı sonucu KAYBETMEMELİ
                print(f"[replay_proofs] UYARI: cleanup_work_dir başarısız: {cleanup_error}", flush=True)
        try:
            write_json_file(result, args.result_out)
            print(f"[replay_proofs] Sonuç yazıldı: {args.result_out}", flush=True)
        except Exception as write_error:  # noqa: BLE001 - son çare: en azından STDOUT'ta görünsün (ebeveyn stdout'u yakalıyor)
            print(f"[replay_proofs] KRİTİK: result.json YAZILAMADI ({write_error}). Ham sonuç:", flush=True)
            print(repr(result), flush=True)
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
    run_id: str,
    segment_index: int,
    verbose: bool = False,
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
        # İşçi süreç result.json YAZMADAN sonlandı — Faz E'nin ilk Colab
        # koşumunda gerçekten yaşandı, HİÇ hata mesajı görünmüyordu. Artık
        # ayrı bir durum ("crashed") + returncode + son 50 satır loglanıyor,
        # ve aşağıdaki verbose/hata bloğu bu durumda da TAM çıktıyı basıyor.
        tail_lines = (stdout or "").splitlines()[-50:]
        result = {
            "combo": {"round_id": round_id, "site_index": site_index},
            "status": "crashed",
            "error_summary": f"Worker süreci result.json yazmadan sonlandı (returncode={proc.returncode}).",
            "stdout_tail": tail_lines,
            "timings": {},
            "peak_rss_kb": None,
        }

    if timed_out:
        result["status"] = "failed"
        result["error_summary"] = f"{result.get('error_summary') or ''} [TIMEOUT: {timeout}s aşıldı]".strip()

    result.setdefault("returncode", proc.returncode)
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

    # `run_id`/`segment_index` işçi süreç HİÇ BİLMEDİĞİ için `result.json`'a
    # (results/{key}.json) YAZILMAMIŞTI — sadece ebeveynin BELLEKTEKİ
    # `all_results` sözlüğüne (ve oradan `replay_results_<mod>.json`'a)
    # ekleniyordu. Faz E'nin ikinci Colab koşumunda run_id'nin TÜM
    # kayıtlarda "-" görünmesinin kök sebebi buydu — kullanıcı per-kombinasyon
    # `results/{key}.json` dosyalarını inceliyordu, onlarda bu alanlar HİÇ
    # yoktu. Artık ebeveyn bu değerleri BURADA (dosyaya geri yazmadan önce)
    # ekliyor, iki artifact (`results/{key}.json` ve `replay_results_<mod>.json`)
    # TUTARLI kalıyor.
    result["run_id"] = run_id
    result["segment_index"] = segment_index
    try:
        write_json_file(result, str(result_path))
    except Exception as rewrite_error:  # noqa: BLE001 - aggregate dosya zaten doğru kaydediliyor, bu sadece per-kombinasyon dosyanın tutarlılığı için
        print(f"[replay_proofs] UYARI: '{key}' için result_path yeniden yazılamadı ({rewrite_error}) — aggregate sonuç yine de doğru.")

    # 1) işçinin TAM çıktısını her zaman göster: başarısızsa KOŞULSUZ,
    #    başarılıysa sadece --verbose ile (bench_circuit.py'de bu hiç
    #    yoktu, Faz E'nin ilk koşumunda "hata mesajı hiç basılmadı"
    #    şikayetinin kök sebebiydi).
    if verbose or result["status"] != "success":
        print(f"\n--- '{key}' işçi süreç TAM çıktısı ({log_path}) ---")
        print(stdout or "(çıktı yok)")
        print(f"--- '{key}' çıktı sonu ---")

    # 5) başarısızlıkta komutu TEKRAR göster (elle yeniden koşturmak için
    #    kopyala-yapıştır) — ilk yazdırma yukarıda, koşumdan ÖNCE oldu,
    #    uzun bir çıktının arasında kaybolmuş olabilir.
    if result["status"] != "success":
        print(f"[replay_proofs] '{key}' BAŞARISIZ (durum={result['status']}). Elle yeniden koşturmak için:")
        print(f"  {' '.join(cmd)}")

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
    parser.add_argument("--mode", default=None, choices=["full", "staged"])
    parser.add_argument("--rounds", default=DEFAULT_ROUNDS)
    parser.add_argument("--sites", default=DEFAULT_SITES)
    parser.add_argument("--only-first", action="store_true", help="sadece ilk kombinasyonu koş")
    parser.add_argument("--limit", type=int, default=None, help="en fazla N ispat koştur (kalanını atla)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="ispat başına saniye")
    parser.add_argument("--force", action="store_true", help="zaten sonuçlandırılmış kombinasyonları yeniden çalıştır")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="başarılı kombinasyonların işçi çıktısını da göster (başarısız olanlar zaten HER ZAMAN gösterilir)")
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


# Sadece `--worker` alt sürecinin GERÇEKTEN kullandığı, ebeveyn modda
# hiçbir anlamı olmayan alanlar (`--shards-dir`/`--circuit-config`/
# `--keep-artifacts` HARİÇ — bunlar HER İKİ modda da kullanılıyor).
WORKER_ONLY_ARG_NAMES = (
    "round_id", "site_index", "challenge_seed_hex", "rpc_url",
    "round_manager_address", "round_manager_abi_file", "private_key",
    "work_dir", "result_out",
)


def _flag_name(attr_name: str) -> str:
    return "--" + attr_name.replace("_", "-")


def validate_args(args: argparse.Namespace) -> None:
    """`parse_args()` `--mode`'u DA, işçiye özgü alanları DA `required=True`
    YAPMAZ — çünkü ebeveyn ve işçi (`--worker`) modlarının ZORUNLU alan
    kümeleri FARKLI ve argparse TEK bir zorunlu-alan kümesi kabul ediyor.
    Faz E'nin ilk Colab koşumunda `--mode` argparse seviyesinde
    `required=True` OLDUĞU için ebeveynin başlattığı `--worker` çağrısı
    (`--mode` GÖNDERMİYOR, worker onu hiç kullanmıyor) doğrudan
    `argparse.error` ile patlıyordu. Bu fonksiyon HER MOD için doğru
    zorunlulukları, argparse'tan SONRA, çalışma zamanında denetler —
    `main()` bunu `run_worker`/ebeveyn dallarına ayrılmadan HEMEN ÖNCE
    çağırır."""
    if args.worker:
        missing = [_flag_name(name) for name in WORKER_ONLY_ARG_NAMES if getattr(args, name) is None]
        if missing:
            raise ValueError(f"--worker ile şunlar da ZORUNLU: {missing}")
        return

    if args.mode is None:
        raise ValueError("--mode ZORUNLU (ebeveyn modda) — 'full' ya da 'staged' seç.")

    ignored = [_flag_name(name) for name in WORKER_ONLY_ARG_NAMES if getattr(args, name, None) is not None]
    if ignored:
        print(f"[replay_proofs] NOT: {ignored} sadece --worker modunda anlamlı, ebeveyn modda YOKSAYILIYOR.")


def main(argv=None) -> int:
    args = parse_args(argv)
    validate_args(args)

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
    infra = get_replay_infra_config(schedule_config)

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
    print(
        f"[replay_proofs] anvil_restart_interval={infra['anvil_restart_interval']} "
        f"rpc_timeout={infra['rpc_timeout_seconds']}s health_check_timeout={infra['health_check_timeout_seconds']}s "
        f"(bkz. configs/schedule.yaml: replay_infra)"
    )

    assert_writable(str(replay_dir))
    replay_dir.mkdir(parents=True, exist_ok=True)
    assert_writable(str(work_root))
    work_root.mkdir(parents=True, exist_ok=True)

    results_path = str(replay_dir / f"replay_results_{args.mode}.json")
    all_results = load_replay_results(results_path)

    # Faz E'nin ilk tam mod koşumunda anvil ÇÖKMESİ yaşanmıştı (36 ispat
    # sonrası) — ham veri incelemesi bunun ezkl'i de yavaşlattığını
    # gösterdi (bkz. classify_environment_health docstring'i). `run_id`
    # HER `main()` çağrısında YENİDEN üretilir (aynı process/oturum
    # boyunca sabit) — bu sayede "hangi kayıt HANGİ Colab koşumundan
    # geldiği" ARTIK her yeni sonuca yazılıyor; ESKİ kayıtlarda yok,
    # onlar timings üzerinden (classify_environment_health) sınıflandırılır.
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_pid{os.getpid()}"
    print(f"[replay_proofs] run_id={run_id}")

    reputations = {s: schedule_config["reputation_initial"] for s in sites}
    processed = 0
    infra_events: list = []  # her yeniden başlatma/sağlık-kontrolü olayının GERÇEK kaydı - raporda GİZLENMEZ

    def open_segment(segment_index: int) -> dict:
        print(f"\n[replay_proofs] === Segment {segment_index}: yeni anvil başlatılıyor ===")
        anvil = AnvilProcess()
        anvil.__enter__()
        if len(anvil.accounts) < len(sites) + 1:
            anvil.__exit__(None, None, None)
            raise RuntimeError(f"anvil yeterli hesap üretmedi: {len(anvil.accounts)} hesap, {len(sites) + 1} gerekli.")
        owner_address, owner_key = anvil.accounts[0]
        site_accounts = {s: anvil.accounts[i + 1] for i, s in enumerate(sites)}

        deploy_client = Web3Client(anvil.rpc_url, timeout=infra["rpc_timeout_seconds"])
        print(f"[replay_proofs] Segment {segment_index}: '{args.round_manager_sol}' derleniyor...")
        compiled_rm = compile_with_fallback_strategies(Path(args.round_manager_sol))
        proof_schedule = schedule_config["proof_schedule"]
        round_manager_address, deploy_rm_gas = deploy_client.deploy_contract(
            compiled_rm["abi"],
            compiled_rm["bytecode"],
            (schedule_config["reputation_initial"], schedule_config["reputation_penalty"], schedule_config["reputation_bonus"], proof_schedule["reputation_threshold"]),
            owner_key,
        )
        print(f"[replay_proofs] Segment {segment_index}: RoundManager deploy edildi: {round_manager_address} (gas={deploy_rm_gas})")

        round_manager_client = RoundManagerClient(anvil.rpc_url, round_manager_address, compiled_rm["abi"], timeout=infra["rpc_timeout_seconds"])
        for s in sites:
            round_manager_client.register_site(site_accounts[s][0], owner_key)
        print(
            f"[replay_proofs] Segment {segment_index}: {len(sites)} site kaydedildi — İTİBARLAR SIFIRLANDI "
            f"(yeni kontrat, hepsi reputation_initial={schedule_config['reputation_initial']}'dan başlıyor). "
            f"'full' modda etkisi yok (itibar hiç kullanılmıyor); 'staged' modda takvim kararı ('must_prove') "
            f"hâlâ bu script'in YEREL `reputations` sözlüğüne dayanıyor (zincir-tarafı sıfırlanmasından bağımsız "
            f"olarak devam ediyor) — ama zincirdeki GERÇEK `isSiteEligible`/`reputation` değeri artık bu segmentin "
            f"başlangıcından itibaren yeniden sayılıyor. Bu, raporda 'altyapı kısıtı' olarak NOT DÜŞÜLÜYOR."
        )
        abi_file = work_root / f"round_manager_abi_segment{segment_index}.json"
        write_json_file(compiled_rm["abi"], str(abi_file))

        return {
            "segment_index": segment_index,
            "anvil": anvil,
            "owner_key": owner_key,
            "site_accounts": site_accounts,
            "round_manager_client": round_manager_client,
            "round_manager_address": round_manager_address,
            "abi_file": abi_file,
        }

    def close_segment(segment: dict) -> None:
        segment["anvil"].__exit__(None, None, None)

    segment = open_segment(0)
    proofs_in_segment = 0

    def restart_segment(reason: str, **context) -> None:
        nonlocal segment, proofs_in_segment
        infra_events.append({"type": reason, "segment_index": segment["segment_index"], **context})
        print(f"[replay_proofs] ANVİL YENİDEN BAŞLATILIYOR (sebep={reason}, bağlam={context}).")
        close_segment(segment)
        segment = open_segment(segment["segment_index"] + 1)
        proofs_in_segment = 0

    try:
        for round_id in rounds:
            if args.limit is not None and processed >= args.limit:
                print(f"[replay_proofs] --limit {args.limit} doldu, durduruluyor.")
                break

            if round_fully_done(round_id, sites, all_results, args.force):
                print(f"[replay_proofs] round={round_id} zaten TAMAMEN sonuçlandırılmış — startRound bile atlanıyor.")
                continue

            if should_restart_segment(proofs_in_segment, infra["anvil_restart_interval"]):
                restart_segment("scheduled_restart", before_round=round_id, proofs_in_segment=proofs_in_segment)

            # Round başlatmadan önce sağlık kontrolü + gerekirse yeniden başlatıp bir kez daha dene.
            # Ana döngüde anvil hatası artık ÖLÜMCÜL DEĞİL: iki deneme de başarısız olursa bu round'un
            # TÜM site'ları "failed" işaretlenip bir SONRAKİ round'a geçiliyor, koşu düşmüyor.
            challenge_seed = None
            for attempt in range(2):
                if not check_rpc_alive(segment["anvil"].rpc_url, timeout=infra["health_check_timeout_seconds"]):
                    restart_segment("unhealthy_before_round", round_id=round_id, attempt=attempt)
                try:
                    challenge_seed, start_gas = segment["round_manager_client"].start_round(
                        round_id, f"replay-round-{round_id}", (round_id).to_bytes(32, "big"), segment["owner_key"]
                    )
                    break
                except Exception as e:  # noqa: BLE001 - bir deneme daha yapılacak, ölümcül DEĞİL
                    print(f"[replay_proofs] round={round_id} startRound BAŞARISIZ (deneme {attempt + 1}/2): {type(e).__name__}: {e}")
                    restart_segment("start_round_failed", round_id=round_id, attempt=attempt, error=f"{type(e).__name__}: {e}")

            if challenge_seed is None:
                print(f"[replay_proofs] round={round_id}: startRound İKİ denemede de başarısız — TÜM site'lar 'failed' işaretlenip sonraki round'a geçiliyor.")
                for site_index in sites:
                    key = replay_combo_key(round_id, site_index)
                    if key in all_results and not args.force:
                        continue
                    all_results[key] = {
                        "combo": {"round_id": round_id, "site_index": site_index},
                        "status": "failed",
                        "error_summary": "startRound iki denemede de başarısız oldu (anvil altyapı sorunu) — bu round'daki hiçbir site ispatlanamadı.",
                        "segment_index": segment["segment_index"],
                        "run_id": run_id,
                    }
                save_replay_results(all_results, results_path)
                continue

            print(f"[replay_proofs] round={round_id} startRound: gas={start_gas} challenge_seed={challenge_seed.hex()} segment={segment['segment_index']}")

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

                # Her ispattan ÖNCE hızlı sağlık kontrolü. Anvil ölmüşse yeniden başlatılıyor —
                # bu round'un ÖNCEKİ site'ları farklı bir challenge_seed ile ispatlanmış olabilir
                # (yeni segment yeni bir startRound gerektirir); bu bilinen bir tutarsızlık, infra_events'e
                # kaydedilip raporda AÇIKÇA belirtiliyor (gizlenmiyor).
                if not check_rpc_alive(segment["anvil"].rpc_url, timeout=infra["health_check_timeout_seconds"]):
                    restart_segment("unhealthy_before_proof", round_id=round_id, site_index=site_index)
                    try:
                        challenge_seed, _start_gas = segment["round_manager_client"].start_round(
                            round_id, f"replay-round-{round_id}-resumed", (round_id).to_bytes(32, "big"), segment["owner_key"]
                        )
                        print(
                            f"[replay_proofs] round={round_id} YENİ segment {segment['segment_index']} için startRound "
                            f"TEKRAR çağrıldı — YENİ challenge_seed={challenge_seed.hex()} (bu round'un daha önce "
                            f"işlenmiş site'larından FARKLI olabilir, bkz. infra_events)."
                        )
                    except Exception as e:  # noqa: BLE001 - bu site için başarısız işaretlenip devam ediliyor
                        all_results[key] = {
                            "combo": {"round_id": round_id, "site_index": site_index},
                            "status": "failed",
                            "error_summary": f"anvil yeniden başlatma sonrası startRound (resume) başarısız: {type(e).__name__}: {e}",
                            "segment_index": segment["segment_index"],
                            "run_id": run_id,
                        }
                        save_replay_results(all_results, results_path)
                        continue

                site_address, site_key = segment["site_accounts"][site_index]

                try:
                    shard = load_shard(shards_dir, round_id, site_index)
                    weight_commitment = compute_weight_commitment(shard)
                    del shard
                    gc.collect()
                    segment["round_manager_client"].submit_update(round_id, f"replay-update-{round_id}-{site_index}", weight_commitment, site_key)
                except Exception as e:  # noqa: BLE001 - bu (round,site) başarısız işaretlenip devam ediliyor
                    result = {
                        "combo": {"round_id": round_id, "site_index": site_index},
                        "status": "failed",
                        "error_summary": f"submitUpdate/shard öncesi hazırlık başarısız: {type(e).__name__}: {e}",
                        "run_id": run_id,
                    }
                    print(f"[replay_proofs] '{key}' BAŞARISIZ (submitUpdate öncesi): {result['error_summary']}")
                else:
                    result = run_combo_in_subprocess(
                        round_id, site_index, challenge_seed,
                        rpc_url=segment["anvil"].rpc_url, round_manager_address=segment["round_manager_address"], abi_file=segment["abi_file"],
                        private_key=site_key, shards_dir=shards_dir, circuit_config_path=args.circuit_config,
                        work_root=work_root, replay_dir=replay_dir, timeout=args.timeout, keep_artifacts=args.keep_artifacts,
                        run_id=run_id, segment_index=segment["segment_index"], verbose=args.verbose,
                    )
                    proofs_in_segment += 1
                    if result.get("verified"):
                        reputations[site_index] = segment["round_manager_client"].get_reputation(site_address)

                result["segment_index"] = segment["segment_index"]
                result["run_id"] = run_id
                all_results[key] = result
                save_replay_results(all_results, results_path)
                processed += 1
                print(
                    f"[replay_proofs] '{key}' bitti: durum={result['status']} segment={segment['segment_index']} "
                    f"tepe_RAM_KB={result.get('peak_rss_kb')} verified={result.get('verified')}"
                )
                if result["status"] != "success":
                    print(f"[replay_proofs]   hata: {result.get('error_summary') or result.get('error')}")
                gc.collect()
    finally:
        close_segment(segment)

    print(f"\n[replay_proofs] Altyapı olayları (yeniden başlatma/sağlık kontrolü) — {len(infra_events)} olay:")
    for event in infra_events:
        print(f"  {event}")

    print(f"\n=== Faz E ({args.mode}) sonuçları ===")
    totals = compute_mode_totals(all_results)
    print(json.dumps(totals, indent=2, ensure_ascii=False))

    other_mode = "staged" if args.mode == "full" else "full"
    other_results_path = replay_dir / f"replay_results_{other_mode}.json"
    other_results: dict = {}
    totals_by_mode = {args.mode: totals}
    if other_results_path.is_file():
        other_results = load_replay_results(str(other_results_path))
        totals_by_mode[other_mode] = compute_mode_totals(other_results)
    else:
        print(f"[replay_proofs] '{other_mode}' modu henüz koşulmadı — karşılaştırma tablosu bu mod için EKSİK kalacak.")

    ordered = {m: totals_by_mode[m] for m in ("full", "staged") if m in totals_by_mode}
    comparison_table = render_mode_comparison_table(ordered)
    print("\n" + comparison_table)

    md_parts = [f"# Faz E — İspat Takvimi Maliyet Karşılaştırması\n\n", "## Mod karşılaştırma tablosu (ham toplamlar)\n\n", comparison_table]

    # Ham toplamlar, anvil'in bellek şişmesi sırasında ölçülmüş (bkz.
    # classify_environment_health) "degraded" ispatları İÇEREBİLİR —
    # bu, özellikle tam mod koşusunda (ilk çöken koşum) toplam ezkl
    # süresini şişirir. Normalize tablo SADECE "healthy" + başarılı
    # ispatların GERÇEK ortalamasını kullanarak bu şişmeyi düzeltir
    # (74s gibi bir sayı UYDURULMUYOR — mevcut healthy verilerden
    # HESAPLANIYOR). Her iki modun BİRLEŞİK sonuçlarından hesaplanır —
    # tek bir modun healthy örneklemi azsa diğer modun healthy verisiyle
    # desteklenir.
    combined_results_for_health = dict(all_results)
    combined_results_for_health.update(other_results)
    healthy_avg_ezkl_seconds = compute_healthy_avg_ezkl_seconds(combined_results_for_health)
    health_buckets = split_by_environment_health(combined_results_for_health)
    degraded_count, unknown_count = len(health_buckets["degraded"]), len(health_buckets["unknown"])

    if healthy_avg_ezkl_seconds is not None:
        print(
            f"[replay_proofs] sağlıklı ortamda ölçülen ispat başına ortalama ezkl süresi: "
            f"{healthy_avg_ezkl_seconds:.2f}s ({degraded_count} bozuk-ortam, {unknown_count} sınıflandırılamayan "
            f"kayıt normalize tablodan HARİÇ tutuldu)"
        )
        normalized_table = render_normalized_comparison_table(ordered, healthy_avg_ezkl_seconds)
        print("\n" + normalized_table)
        md_parts += [
            "\n## Normalize edilmiş karşılaştırma (sadece sağlıklı ortamda ölçülen ispatlar)\n\n",
            f"Sağlıklı ortamda ölçülen ispat başına ortalama ezkl süresi: **{healthy_avg_ezkl_seconds:.2f}s** "
            f"— {degraded_count} bozuk-ortam ({ENVIRONMENT_DEGRADED_THRESHOLD_SECONDS:.0f}s'yi aşan setup+prove, "
            f"anvil'in bellek şişmesi sırasında ölçülmüş) ve {unknown_count} sınıflandırılamayan (başarısız/eksik "
            f"timings) kayıt bu ortalamadan HARİÇ tutuldu — bkz. `docs/phase_e_report.md`.\n\n",
            normalized_table,
        ]
    else:
        print("[replay_proofs] UYARI: hiç 'healthy' ortamda ölçülmüş başarılı ispat yok — normalize tablo ÜRETİLEMİYOR.")
        md_parts += ["\n## Normalize edilmiş karşılaştırma\n\n", "Hiç 'healthy' ortamda ölçülmüş başarılı ispat yok — normalize tablo üretilemedi.\n"]
    if "staged" in totals_by_mode:
        staged_results = all_results if args.mode == "staged" else load_replay_results(str(replay_dir / "replay_results_staged.json"))
        md_parts += ["\n## \"staged\" modda round başına ispat dağılımı\n\n", render_staged_distribution(staged_results)]

    failure_summary = render_failure_summary(all_results)
    print(f"\n=== '{args.mode}' modunda başarısız/çöken kombinasyonlar ===")
    print(failure_summary)
    md_parts += [f"\n## \"{args.mode}\" modunda başarısız/çöken kombinasyonlar\n\n", failure_summary]

    # infra_events BİLEREK all_results'ın İÇİNE konmuyor (compute_mode_totals/
    # render_failure_summary/round_fully_done onu bir "kombinasyon" sanıp
    # patlardı) — ayrı, küçük bir dosyaya yazılıyor. "Kesinti ve yeniden
    # başlatma sayısı" raporda GİZLENMEZ (kullanıcının açık isteği).
    infra_summary = render_infra_events(infra_events)
    print(f"\n=== '{args.mode}' modunda altyapı olayları (yeniden başlatma/sağlık kontrolü) ===")
    print(infra_summary)
    md_parts += [f"\n## \"{args.mode}\" modunda altyapı olayları (yeniden başlatma/sağlık kontrolü)\n\n", infra_summary]

    infra_events_path = str(replay_dir / f"replay_infra_events_{args.mode}.json")
    write_json_file({"infra_events": infra_events, "restart_count": len(infra_events)}, infra_events_path)
    print(f"[replay_proofs] Yazıldı: {infra_events_path}")

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
