#!/usr/bin/env python
"""Faz A3: fedavg_N.pt numaralandırmasını denetle, delta-G norm istatistiklerini çıkar.

BİLİNÇLİ TEST SINIRI: `--dry-run` dışındaki akış gerçek StyleGAN-XL
pickle'ları, gerçek `fedavg_*.pt` dosyaları ve gerçek bir StyleGAN-XL
klonu gerektirir. Yerelde hiçbiri yok, bu yüzden `main()`'in birim
testi yoktur. Saf mantık (fl/fedavg_utils.py) ayrı test edilmiştir.

Kullanım (Colab'da):
    python -m scripts.audit_fedavg --dry-run
    python -m scripts.audit_fedavg
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict

import torch

from configs.loader import load_paths
from fl.fedavg_utils import RunningAverage, compare_state_dicts, compute_norm_percentiles
from fl.stylegan_xl_env import load_network_pkl
from scripts.inventory import EXPECTED_SITES, scan_structure


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Faz A3: fedavg numaralandırma denetimi + delta-G norm istatistikleri.")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--stylegan-xl-repo", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Hangi round'lar denetlenebilir, sadece raporla, pkl açma.")
    return parser.parse_args(argv)


def resolve_paths(args) -> tuple[str, str, str]:
    paths = load_paths(env=args.env, config_path=args.paths_config)
    raw_root = args.raw_root or paths["raw_root"]
    results_dir = args.results_dir or paths["results_dir"]
    stylegan_xl_repo = args.stylegan_xl_repo or paths["stylegan_xl_repo"]
    return raw_root, results_dir, stylegan_xl_repo


def group_sites_by_round(structure: dict) -> dict[int, list[dict]]:
    by_round: dict[int, list[dict]] = defaultdict(list)
    for site in structure["sites"]:
        if site["files"].get("network-snapshot.pkl"):
            by_round[site["round"]].append(site)
    return by_round


def load_full_state_dict(pkl_path: str, stylegan_xl_repo: str) -> dict:
    data = load_network_pkl(pkl_path, stylegan_xl_repo)
    g_ema = data["G_ema"]
    state = {k: v.detach().cpu().to(torch.float32).clone() for k, v in g_ema.state_dict().items()}
    del data, g_ema
    gc.collect()
    return state


def load_fedavg_file(path: str) -> dict:
    raw = torch.load(path, map_location="cpu")
    return {k: v.detach().cpu().to(torch.float32) for k, v in raw.items()}


def audit_round(round_idx: int, sites: list[dict], raw_root: str, stylegan_xl_repo: str) -> tuple[dict, list[float]]:
    print(f"\n=== round {round_idx}: {len(sites)} site ortalanıyor ===")
    avg = RunningAverage()
    for site in sorted(sites, key=lambda s: s["site"]):
        pkl_path = site["files"]["network-snapshot.pkl"]
        print(f"[audit_fedavg] Yükleniyor: {pkl_path}")
        state = load_full_state_dict(pkl_path, stylegan_xl_repo)
        avg.update(state)
        del state
        gc.collect()

    computed_avg = avg.result()
    round_result: dict = {"round": round_idx, "num_sites_averaged": avg.count}

    for candidate_n in (round_idx, round_idx + 1):
        fedavg_path = os.path.join(raw_root, f"fedavg_{candidate_n}.pt")
        if not os.path.isfile(fedavg_path):
            print(f"[audit_fedavg] fedavg_{candidate_n}.pt bulunamadı, karşılaştırma atlanıyor.")
            continue
        fedavg_state = load_fedavg_file(fedavg_path)
        cmp = compare_state_dicts(computed_avg, fedavg_state)
        print(
            f"[audit_fedavg] round {round_idx} hesaplanan ortalama vs fedavg_{candidate_n}.pt: "
            f"overall_max_abs_diff={cmp['overall_max_abs_diff']}, "
            f"only_in_computed={len(cmp['only_in_a'])}, only_in_file={len(cmp['only_in_b'])}"
        )
        round_result[f"vs_fedavg_{candidate_n}"] = {
            "overall_max_abs_diff": cmp["overall_max_abs_diff"],
            "only_in_computed": cmp["only_in_a"],
            "only_in_fedavg_file": cmp["only_in_b"],
        }
        del fedavg_state
        gc.collect()

    deltas: list[float] = []
    prev_global_path = os.path.join(raw_root, f"fedavg_{round_idx - 1}.pt") if round_idx > 0 else None
    if not prev_global_path or not os.path.isfile(prev_global_path):
        print(
            f"[audit_fedavg] round {round_idx}: önceki global (fedavg_{round_idx - 1}.pt) yok, "
            f"bu round için delta-G normu hesaplanmıyor."
        )
        return round_result, deltas

    prev_global = load_fedavg_file(prev_global_path)
    for site in sorted(sites, key=lambda s: s["site"]):
        pkl_path = site["files"]["network-snapshot.pkl"]
        site_state = load_full_state_dict(pkl_path, stylegan_xl_repo)
        common_keys = set(site_state) & set(prev_global)
        if not common_keys:
            print(f"UYARI: round {round_idx} site {site['site']}: prev_global ile ortak anahtar yok, delta-G normu atlanıyor.")
            del site_state
            gc.collect()
            continue

        sq_sum = 0.0
        for key in common_keys:
            diff = (site_state[key] - prev_global[key]).flatten()
            sq_sum += torch.dot(diff, diff).item()
        norm = sq_sum**0.5
        deltas.append(norm)
        print(f"[audit_fedavg] round {round_idx} site {site['site']}: ||delta-G|| = {norm:.6f}")
        del site_state
        gc.collect()

    del prev_global
    gc.collect()
    return round_result, deltas


def main(argv=None) -> int:
    args = parse_args(argv)
    raw_root, results_dir, stylegan_xl_repo = resolve_paths(args)

    print(f"[audit_fedavg] raw_root={raw_root}")
    print(f"[audit_fedavg] results_dir={results_dir}")
    print(f"[audit_fedavg] dry_run={args.dry_run}")

    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"raw_root bulunamadı: '{raw_root}'.")

    structure = scan_structure(raw_root)
    by_round = group_sites_by_round(structure)

    complete_rounds = sorted(r for r, sites in by_round.items() if len(sites) == EXPECTED_SITES)
    incomplete_rounds = sorted(r for r, sites in by_round.items() if len(sites) != EXPECTED_SITES)
    if incomplete_rounds:
        print(f"UYARI: şu round'larda {EXPECTED_SITES} site pkl'ı yok, denetlenemeyecek: {incomplete_rounds}")

    if args.dry_run:
        print(f"\n[audit_fedavg] Denetlenebilecek round'lar: {complete_rounds}")
        print(f"[audit_fedavg] fedavg dosyaları: {structure['fedavg_files']}")
        return 0

    os.makedirs(results_dir, exist_ok=True)

    per_round_results = []
    all_deltas: list[float] = []
    for round_idx in complete_rounds:
        round_result, deltas = audit_round(round_idx, by_round[round_idx], raw_root, stylegan_xl_repo)
        per_round_results.append(round_result)
        all_deltas.extend(deltas)

    fedavg_audit_path = os.path.join(results_dir, "fedavg_audit.json")
    with open(fedavg_audit_path, "w", encoding="utf-8") as f:
        json.dump(
            {"complete_rounds": complete_rounds, "incomplete_rounds": incomplete_rounds, "per_round": per_round_results},
            f,
            indent=2,
            default=str,
            ensure_ascii=False,
        )
    print(f"\n[audit_fedavg] Yazıldı: {fedavg_audit_path}")

    if not all_deltas:
        raise RuntimeError(
            "Hiçbir round için delta-G normu hesaplanamadı (muhtemelen önceki global fedavg dosyaları "
            "eksik). delta_norms.json uydurulmuyor, script hata ile duruyor."
        )

    percentiles = compute_norm_percentiles(all_deltas)
    delta_norms_path = os.path.join(results_dir, "delta_norms.json")
    with open(delta_norms_path, "w", encoding="utf-8") as f:
        json.dump({"deltas": all_deltas, "percentiles": percentiles}, f, indent=2, ensure_ascii=False)
    print(f"[audit_fedavg] Yazıldı: {delta_norms_path} (p50={percentiles['p50']:.4f}, p90={percentiles['p90']:.4f}, p99={percentiles['p99']:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
