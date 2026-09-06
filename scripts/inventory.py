#!/usr/bin/env python
"""Faz A1: parallel_final envanteri.

BİLİNÇLİ TEST SINIRI: Bu script'in `--dry-run` DIŞINDAKİ akışı gerçek
StyleGAN-XL pickle dosyaları ve gerçek bir StyleGAN-XL klonu gerektirir.
Yerelde ikisi de yok; bu yüzden `main()`'in birim testi yoktur. Saf
mantık (fl/module_tree.py, fl/inventory_utils.py) ayrı test edilmiştir.

Kullanım (Colab'da):
    python -m scripts.inventory --dry-run
    python -m scripts.inventory
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import re

from configs.loader import load_paths
from fl.inventory_utils import classify_pkl_paths, compare_training_options, parse_stats_jsonl
from fl.module_tree import describe_module_tree, detect_candidate_submodules, extract_known_attrs
from fl.stylegan_xl_env import load_network_pkl
from storage.pathguard import assert_writable

EXPECTED_ROUNDS = 15  # round_0..round_14
EXPECTED_SITES = 4  # site_0..site_3
EXPECTED_FILES = ("log.txt", "stats.jsonl", "training_options.json")
SITE_DIR_PATTERN = re.compile(r"^site_[0-3]$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Faz A1: parallel_final envanteri çıkar.")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--stylegan-xl-repo", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hiçbir pkl/json dosyası açmadan sadece dizin yapısını tara ve raporla.",
    )
    return parser.parse_args(argv)


def resolve_paths(args) -> tuple[str, str, str]:
    paths = load_paths(env=args.env, config_path=args.paths_config)
    raw_root = args.raw_root or paths["raw_root"]
    results_dir = args.results_dir or paths["results_dir"]
    stylegan_xl_repo = args.stylegan_xl_repo or paths["stylegan_xl_repo"]
    return raw_root, results_dir, stylegan_xl_repo


def find_round_site_dirs(raw_root: str) -> list[tuple[int, int, str]]:
    pattern = os.path.join(raw_root, "round_*", "site_*")
    dirs = sorted(glob.glob(pattern))
    entries = []
    for d in dirs:
        parts = d.replace("\\", "/").split("/")
        round_part = next((p for p in parts if p.startswith("round_")), None)
        site_part = next((p for p in parts if p.startswith("site_")), None)
        if round_part is None or site_part is None:
            print(f"UYARI: beklenmeyen dizin adı, round/site çıkarılamadı: {d}")
            continue

        if not SITE_DIR_PATTERN.match(site_part):
            print(
                f"[inventory] Yoksayıldı: '{d}' — site dizini adı beklenen desene "
                f"(^site_[0-3]$) uymuyor, bulunan: '{site_part}'."
            )
            continue

        try:
            round_idx = int(round_part.split("_")[1])
            site_idx = int(site_part.split("_")[1])
        except (IndexError, ValueError):
            print(f"UYARI: round/site numarası parse edilemedi: {d}")
            continue
        entries.append((round_idx, site_idx, d))
    return entries


def find_run_subdir(site_dir: str) -> tuple[str | None, str | None]:
    candidates = [d for d in glob.glob(os.path.join(site_dir, "*")) if os.path.isdir(d)]
    if len(candidates) == 0:
        return None, f"'{site_dir}' altında alt çalışma klasörü bulunamadı (ör. 00000-stylegan3-r-site*)."
    if len(candidates) > 1:
        return None, f"'{site_dir}' altında birden fazla alt klasör bulundu, hangisi doğru belirsiz: {candidates}"
    return candidates[0], None


def find_snapshot_pkl(run_dir: str) -> tuple[str | None, str | None]:
    exact = os.path.join(run_dir, "network-snapshot.pkl")
    if os.path.isfile(exact):
        return exact, None

    candidates = sorted(glob.glob(os.path.join(run_dir, "network-snapshot*.pkl")))
    if not candidates:
        return None, f"'{run_dir}' altında network-snapshot*.pkl bulunamadı."
    if len(candidates) > 1:
        chosen = candidates[-1]
        return chosen, (
            f"Tam olarak 'network-snapshot.pkl' yok, {len(candidates)} aday bulundu: "
            f"{candidates}. En sonuncusu seçildi: {chosen}"
        )
    return candidates[0], f"'network-snapshot.pkl' yerine '{candidates[0]}' bulundu."


def scan_structure(raw_root: str) -> dict:
    print(f"[inventory] Dizin yapısı taranıyor: {raw_root}")
    entries = find_round_site_dirs(raw_root)
    print(f"[inventory] {len(entries)} adet round_*/site_* dizini bulundu.")

    sites = []
    found_pairs = set()
    for round_idx, site_idx, site_dir in entries:
        found_pairs.add((round_idx, site_idx))
        run_dir, run_dir_warning = find_run_subdir(site_dir)
        if run_dir_warning:
            print(f"UYARI [round {round_idx} site {site_idx}]: {run_dir_warning}")

        site_report = {
            "round": round_idx,
            "site": site_idx,
            "site_dir": site_dir,
            "run_dir": run_dir,
            "run_dir_warning": run_dir_warning,
            "files": {},
        }

        if run_dir:
            for fname in EXPECTED_FILES:
                fpath = os.path.join(run_dir, fname)
                exists = os.path.isfile(fpath)
                site_report["files"][fname] = fpath if exists else None
                if not exists:
                    print(f"UYARI [round {round_idx} site {site_idx}]: '{fname}' bulunamadı.")

            pkl_path, pkl_warning = find_snapshot_pkl(run_dir)
            site_report["files"]["network-snapshot.pkl"] = pkl_path
            site_report["pkl_warning"] = pkl_warning
            if pkl_warning:
                print(f"UYARI [round {round_idx} site {site_idx}]: {pkl_warning}")

        sites.append(site_report)

    expected_pairs = {(r, s) for r in range(EXPECTED_ROUNDS) for s in range(EXPECTED_SITES)}
    missing = sorted(expected_pairs - found_pairs)
    unexpected = sorted(found_pairs - expected_pairs)
    if missing:
        print(f"UYARI: beklenen {len(expected_pairs)} (round,site) çiftinden {len(missing)} tanesi eksik: {missing}")
    if unexpected:
        print(f"UYARI: beklenmeyen (round,site) çiftleri bulundu (0-{EXPECTED_ROUNDS - 1} / 0-{EXPECTED_SITES - 1} dışında): {unexpected}")

    fedavg_pattern = os.path.join(raw_root, "fedavg_*.pt")
    fedavg_files = sorted(glob.glob(fedavg_pattern))
    print(f"[inventory] {len(fedavg_files)} adet fedavg_*.pt dosyası bulundu: {fedavg_files}")

    return {
        "raw_root": raw_root,
        "sites": sites,
        "missing_expected_pairs": missing,
        "unexpected_pairs": unexpected,
        "fedavg_files": fedavg_files,
    }


def run_full_inventory(structure: dict, stylegan_xl_repo: str) -> dict:
    per_site_results = []
    training_options_by_path: dict = {}
    stats_lines_by_path: dict = {}

    for site in structure["sites"]:
        pkl_path = site["files"].get("network-snapshot.pkl")
        if not pkl_path:
            print(f"UYARI: round {site['round']} site {site['site']} için pkl yok, atlanıyor.")
            per_site_results.append({"round": site["round"], "site": site["site"], "error": "pkl bulunamadı"})
            continue

        print(f"\n=== round {site['round']} site {site['site']}: {pkl_path} ===")
        try:
            data = load_network_pkl(pkl_path, stylegan_xl_repo)
        except Exception as e:  # noqa: BLE001 - tanılama amaçlı, hatayı yutmadan raporluyoruz
            print(f"HATA: pkl yüklenemedi: {e}")
            per_site_results.append({"round": site["round"], "site": site["site"], "error": str(e)})
            continue

        if "G_ema" not in data:
            print(f"UYARI: pkl içinde 'G_ema' anahtarı yok. Bulunan anahtarlar: {sorted(data.keys())}")
            per_site_results.append(
                {
                    "round": site["round"],
                    "site": site["site"],
                    "error": "G_ema anahtari yok",
                    "pkl_keys": sorted(data.keys()),
                }
            )
            del data
            gc.collect()
            continue

        g_ema = data["G_ema"]
        module_tree = describe_module_tree(g_ema)
        candidates = detect_candidate_submodules(g_ema)
        dims = extract_known_attrs(g_ema)

        print(f"[inventory] Modül sayısı: {len(module_tree)}")
        print(f"[inventory] Mapping adayları: {candidates.get('mapping')}")
        print(f"[inventory] Embedding adayları: {candidates.get('embedding')}")
        print(f"[inventory] Boyutlar bulunan: {dims['found']}, bulunamayan: {dims['bulunamadi']}")

        per_site_results.append(
            {
                "round": site["round"],
                "site": site["site"],
                "pkl_path": pkl_path,
                "pkl_top_level_keys": sorted(data.keys()),
                "num_modules": len(module_tree),
                "module_tree": module_tree,
                "candidate_submodules": candidates,
                "dims": dims,
            }
        )

        if site["files"].get("training_options.json"):
            path = site["files"]["training_options.json"]
            with open(path, "r", encoding="utf-8") as f:
                try:
                    training_options_by_path[path] = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"UYARI: '{path}' parse edilemedi: {e}")

        if site["files"].get("stats.jsonl"):
            path = site["files"]["stats.jsonl"]
            with open(path, "r", encoding="utf-8") as f:
                stats_lines_by_path[path] = f.readlines()

        del data, g_ema
        gc.collect()

    training_options_report = (
        compare_training_options(training_options_by_path)
        if training_options_by_path
        else {"num_files": 0, "all_keys": [], "differing_keys": {}}
    )
    if training_options_report["differing_keys"]:
        print(f"UYARI: training_options.json dosyaları arasında farklılık var: {sorted(training_options_report['differing_keys'].keys())}")

    stats_report = {path: parse_stats_jsonl(lines) for path, lines in stats_lines_by_path.items()}

    all_pkls_in_tree = glob.glob(os.path.join(structure["raw_root"], "**", "*.pkl"), recursive=True)
    pkl_classification = classify_pkl_paths(all_pkls_in_tree, structure["raw_root"])
    print(f"[inventory] Base model adayı olabilecek pkl'lar (round/site deseni dışında): {pkl_classification['base_model_candidates']}")

    return {
        "per_site": per_site_results,
        "training_options_report": training_options_report,
        "stats_report": stats_report,
        "pkl_classification": pkl_classification,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    raw_root, results_dir, stylegan_xl_repo = resolve_paths(args)

    print(f"[inventory] raw_root={raw_root}")
    print(f"[inventory] results_dir={results_dir}")
    print(f"[inventory] stylegan_xl_repo={stylegan_xl_repo}")
    print(f"[inventory] dry_run={args.dry_run}")

    if not os.path.isdir(raw_root):
        raise FileNotFoundError(
            f"raw_root bulunamadı: '{raw_root}'. Drive mount edildi mi, --raw-root doğru mu?"
        )

    structure = scan_structure(raw_root)

    inventory: dict = {"structure": structure}

    if args.dry_run:
        print("\n[inventory] --dry-run: pkl/json dosyaları açılmadı, sadece dizin yapısı raporlandı.")
    else:
        inventory["full_scan"] = run_full_inventory(structure, stylegan_xl_repo)

    assert_writable(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "inventory.json")
    assert_writable(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n[inventory] Yazıldı: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
