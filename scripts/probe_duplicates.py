#!/usr/bin/env python
"""Faz A raporu Bölüm D'yi netleştir: shard seviyesinde kopya kontrolü.

Round 14'te 4 sitenin delta-G normu neredeyse birebir aynıydı
(docs/phase_a_report.md, Bölüm D). Bunun genuine yakınsama mı yoksa
bir sitenin önceki round'un global ağırlığını hiç eğitmeden olduğu gibi
kaydetmesi mi olduğunu netleştirmek için: her (round, site) shard'ının
`canonical_hash`'ini hesaplar, (a) aynı round içindeki siteler arasında
birebir aynı hash olup olmadığına, (b) bir sitenin hash'inin bir önceki
round'un global ağırlığıyla (fedavg_(N-1)'in G_ema'sının aynı prefixlerle
filtrelenmiş hali) aynı olup olmadığına bakar.

Sadece shard'lar (mapping + embedding katmanları, ~MB) ve fedavg_*.pt
(G/G_ema state_dict'leri) kullanılır — `network-snapshot.pkl` hiç
açılmaz, StyleGAN-XL reposuna ihtiyaç yoktur. Bu yüzden hızlıdır.

BİLİNÇLİ TEST SINIRI: gerçek shard/meta dosyaları (Drive'daki
zk_artifacts/shards) ve gerçek fedavg_*.pt dosyaları gerektirir,
yerelde test edilemez. Kullandığı alt fonksiyonlar
(storage.hashing.canonical_hash, fl.shard_utils.extract_prefixed_state_dict,
scripts.audit_fedavg.load_fedavg_file) ayrı ayrı zaten test edilmiştir.

Kullanım (Colab'da):
    python -m scripts.probe_duplicates --dry-run
    python -m scripts.probe_duplicates
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict

import torch

from configs.loader import load_paths
from fl.shard_utils import extract_prefixed_state_dict
from scripts.audit_fedavg import load_fedavg_file
from storage.hashing import canonical_hash
from storage.pathguard import assert_writable

EXPECTED_ROUNDS = 15  # round_0..round_14
EXPECTED_SITES = 4  # site_0..site_3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Shard seviyesinde round-içi/round-arası kopya kontrolü (Faz A raporu Bölüm D)."
    )
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--raw-root", default=None, help="fedavg_*.pt dosyalarının bulunduğu kök (parallel_final)")
    parser.add_argument("--shards-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hangi shard/meta/fedavg dosyalarının bulunacağını raporla, hiçbir tensör yüklenmez.",
    )
    return parser.parse_args(argv)


def resolve_paths(args) -> tuple[str, str, str]:
    paths = load_paths(env=args.env, config_path=args.paths_config)
    raw_root = args.raw_root or paths["raw_root"]
    shards_dir = args.shards_dir or paths["shards_dir"]
    results_dir = args.results_dir or paths["results_dir"]
    return raw_root, shards_dir, results_dir


def find_shard_entries(shards_dir: str) -> list[tuple[int, int, str, str]]:
    entries = []
    for round_idx in range(EXPECTED_ROUNDS):
        round_dir = os.path.join(shards_dir, f"round_{round_idx}")
        if not os.path.isdir(round_dir):
            print(f"UYARI: '{round_dir}' yok, round {round_idx} tamamen atlanıyor.")
            continue
        for site_idx in range(EXPECTED_SITES):
            shard_path = os.path.join(round_dir, f"site_{site_idx}.pt")
            meta_path = os.path.join(round_dir, f"site_{site_idx}_meta.json")
            if not os.path.isfile(shard_path):
                print(f"UYARI: round {round_idx} site {site_idx}: shard yok ('{shard_path}'), atlanıyor.")
                continue
            if not os.path.isfile(meta_path):
                print(f"UYARI: round {round_idx} site {site_idx}: meta.json yok ('{meta_path}'), atlanıyor.")
                continue
            entries.append((round_idx, site_idx, shard_path, meta_path))
    return entries


def load_shard_hash(shard_path: str) -> str:
    # Shard'lar scripts/extract_shards.py tarafından düz {ad: tensör}
    # olarak yazıldı (özel sınıf/nesne yok), weights_only=True güvenle kullanılabilir.
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    shard = {k: v.detach().cpu().to(torch.float32) for k, v in shard.items()}
    return canonical_hash(shard)


def load_prev_global_shard_hash(raw_root: str, round_idx: int, prefixes: list[str]) -> tuple[str | None, str | None]:
    """fedavg_(round_idx-1).pt'nin G_ema'sını aynı prefixlerle filtreleyip hash'ler.

    round_idx == 0 ise önceki global yok -> (None, None).
    """
    if round_idx == 0:
        return None, None

    prev_path = os.path.join(raw_root, f"fedavg_{round_idx - 1}.pt")
    if not os.path.isfile(prev_path):
        print(f"UYARI: '{prev_path}' bulunamadı, round {round_idx} için önceki global karşılaştırması atlanıyor.")
        return None, None

    prev_g_ema = load_fedavg_file(prev_path)
    prev_shard = extract_prefixed_state_dict(prev_g_ema, prefixes)
    prev_hash = canonical_hash(prev_shard)
    del prev_g_ema, prev_shard
    gc.collect()
    return prev_hash, prev_path


def print_dry_run_report(entries: list[tuple[int, int, str, str]], raw_root: str) -> None:
    by_round: dict[int, list[int]] = defaultdict(list)
    for round_idx, site_idx, _, _ in entries:
        by_round[round_idx].append(site_idx)

    for round_idx in range(EXPECTED_ROUNDS):
        sites = sorted(by_round.get(round_idx, []))
        prev_fedavg = os.path.join(raw_root, f"fedavg_{round_idx - 1}.pt") if round_idx > 0 else None
        prev_status = "yok (round 0)" if round_idx == 0 else ("var" if os.path.isfile(prev_fedavg) else "BULUNAMADI")
        print(f"round {round_idx:>2}: shard bulunan siteler={sites} ({len(sites)}/{EXPECTED_SITES})  önceki global={prev_status}")


def main(argv=None) -> int:
    args = parse_args(argv)
    raw_root, shards_dir, results_dir = resolve_paths(args)

    print(f"[probe_duplicates] raw_root={raw_root}")
    print(f"[probe_duplicates] shards_dir={shards_dir}")
    print(f"[probe_duplicates] results_dir={results_dir}")
    print(f"[probe_duplicates] dry_run={args.dry_run}")

    if not os.path.isdir(shards_dir):
        raise FileNotFoundError(f"shards_dir bulunamadı: '{shards_dir}'.")

    entries = find_shard_entries(shards_dir)
    print(f"[probe_duplicates] {len(entries)} adet (round,site) shard'ı bulundu (beklenen: {EXPECTED_ROUNDS * EXPECTED_SITES}).")

    if args.dry_run:
        print_dry_run_report(entries, raw_root)
        return 0

    by_round: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for round_idx, site_idx, shard_path, meta_path in entries:
        by_round[round_idx].append((site_idx, shard_path, meta_path))

    per_round_results = []

    for round_idx in sorted(by_round):
        sites = sorted(by_round[round_idx], key=lambda t: t[0])
        print(f"\n=== round {round_idx}: {len(sites)} shard hash'leniyor ===")

        site_hashes: dict[int, str] = {}
        site_prefixes: dict[int, list[str]] = {}
        for site_idx, shard_path, meta_path in sites:
            site_hashes[site_idx] = load_shard_hash(shard_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            site_prefixes[site_idx] = meta.get("prefixes_used", [])
            print(f"[probe_duplicates] round {round_idx} site {site_idx}: hash={site_hashes[site_idx][:16]}...")

        distinct_prefix_sets = {tuple(sorted(p)) for p in site_prefixes.values()}
        if len(distinct_prefix_sets) > 1:
            print(
                f"UYARI: round {round_idx}: siteler arasında farklı 'prefixes_used' bulundu "
                f"({distinct_prefix_sets}) — hash karşılaştırması geçersiz olabilir."
            )

        hash_to_sites: dict[str, list[int]] = defaultdict(list)
        for site_idx, h in site_hashes.items():
            hash_to_sites[h].append(site_idx)
        duplicate_groups = [sorted(v) for v in hash_to_sites.values() if len(v) > 1]
        num_unique_hashes = len(hash_to_sites)

        print(f"[probe_duplicates] round {round_idx}: {num_unique_hashes} benzersiz hash (beklenen: {EXPECTED_SITES}).")
        if duplicate_groups:
            print(f"UYARI: round {round_idx}: birebir aynı hash'e sahip site grupları: {duplicate_groups}")

        prev_global_hash = None
        prev_global_path = None
        prev_global_matches: dict[int, bool] = {}
        first_site_prefixes = next(iter(site_prefixes.values()), [])
        if first_site_prefixes:
            prev_global_hash, prev_global_path = load_prev_global_shard_hash(raw_root, round_idx, first_site_prefixes)

        if prev_global_hash is not None:
            for site_idx, h in site_hashes.items():
                matches = h == prev_global_hash
                prev_global_matches[site_idx] = matches
                if matches:
                    print(
                        f"UYARI: round {round_idx} site {site_idx}: shard hash'i önceki global "
                        f"({prev_global_path}) ile BİREBİR AYNI — bu site hiç eğitilmemiş olabilir."
                    )

        per_round_results.append(
            {
                "round": round_idx,
                "site_hashes": {str(k): v for k, v in site_hashes.items()},
                "num_unique_hashes": num_unique_hashes,
                "duplicate_groups": duplicate_groups,
                "prev_global_path": prev_global_path,
                "prev_global_hash": prev_global_hash,
                "prev_global_matches": {str(k): v for k, v in prev_global_matches.items()},
            }
        )

    print("\n=== Özet tablo ===")
    print(f"{'round':>5} {'#hash':>6} {'aynı(round-içi)':>16} {'global ile aynı':>16}")
    for r in per_round_results:
        dup_str = ",".join("+".join(str(s) for s in g) for g in r["duplicate_groups"]) or "-"
        same_as_global = [s for s, m in r["prev_global_matches"].items() if m]
        same_str = ",".join(same_as_global) or "-"
        print(f"{r['round']:>5} {r['num_unique_hashes']:>6} {dup_str:>16} {same_str:>16}")

    assert_writable(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "duplicate_check.json")
    assert_writable(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_round": per_round_results}, f, indent=2, ensure_ascii=False)
    print(f"\n[probe_duplicates] Yazıldı: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
