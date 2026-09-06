#!/usr/bin/env python
"""Faz A2: mapping + sınıf gömme ağırlıklarını fp32 shard olarak çıkar.

BİLİNÇLİ TEST SINIRI: `--dry-run` dışındaki akış gerçek StyleGAN-XL
pickle'ları ve `scripts/inventory.py`'nin (dry-run OLMADAN) ürettiği
gerçek bir inventory.json gerektirir. Yerelde ikisi de yok, bu yüzden
`main()`'in birim testi yoktur. Saf mantık (fl/shard_utils.py) ayrı
test edilmiştir.

Kullanım (Colab'da, önce scripts.inventory tam modda koşmuş olmalı):
    python -m scripts.extract_shards --dry-run
    python -m scripts.extract_shards
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import torch

from configs.loader import load_paths
from fl.shard_utils import build_shard_meta, extract_prefixed_state_dict
from fl.stylegan_xl_env import load_network_pkl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Faz A2: mapping/embedding shard'larını çıkar.")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--results-dir", default=None, help="inventory.json'un okunacağı yer")
    parser.add_argument("--shards-dir", default=None)
    parser.add_argument("--inventory", default=None, help="varsayılan: {results_dir}/inventory.json")
    parser.add_argument("--stylegan-xl-repo", default=None)
    parser.add_argument("--progress-file", default=None, help="varsayılan: {shards_dir}/progress.json")
    parser.add_argument("--dry-run", action="store_true", help="Hangi shard'ların işleneceğini/atlanacağını raporla, pkl açma.")
    return parser.parse_args(argv)


def resolve_paths(args) -> tuple[str, str, str, str]:
    paths = load_paths(env=args.env, config_path=args.paths_config)
    raw_root = args.raw_root or paths["raw_root"]
    results_dir = args.results_dir or paths["results_dir"]
    shards_dir = args.shards_dir or paths["shards_dir"]
    stylegan_xl_repo = args.stylegan_xl_repo or paths["stylegan_xl_repo"]
    return raw_root, results_dir, shards_dir, stylegan_xl_repo


def load_inventory(inventory_path: str) -> dict:
    if not os.path.isfile(inventory_path):
        raise FileNotFoundError(
            f"inventory.json bulunamadı: '{inventory_path}'. Önce 'python -m scripts.inventory' çalıştır."
        )
    with open(inventory_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_site_entries(inventory: dict) -> list[dict]:
    full_scan = inventory.get("full_scan")
    if full_scan is None:
        raise RuntimeError(
            "inventory.json bir '--dry-run' çıktısı gibi görünüyor (içinde 'full_scan' yok). "
            "Önce 'python -m scripts.inventory' (dry-run OLMADAN) çalıştırılmalı."
        )
    return full_scan["per_site"]


def resolve_prefixes(site_entry: dict) -> list[str]:
    if "error" in site_entry:
        raise RuntimeError(
            f"round {site_entry.get('round')} site {site_entry.get('site')} inventory'de hata "
            f"içeriyor, shard çıkarılamaz: {site_entry['error']}"
        )

    candidates = site_entry.get("candidate_submodules", {})
    mapping = candidates.get("mapping", [])
    embedding = candidates.get("embedding", [])

    if not mapping and not embedding:
        raise RuntimeError(
            f"round {site_entry['round']} site {site_entry['site']}: inventory.json'da "
            f"mapping/embedding alt modülü tespit edilmemiş. Uydurma yapılmıyor — "
            f"inventory.json'daki 'module_tree'ye bakıp fl/module_tree.py: DEFAULT_KEYWORD_GROUPS'u "
            f"gerekirse genişlet, sonra inventory.py'yi yeniden çalıştır."
        )

    return sorted(set(mapping) | set(embedding))


def load_progress(progress_path: str) -> dict:
    if os.path.isfile(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress_path: str, progress: dict) -> None:
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, progress_path)


def main(argv=None) -> int:
    args = parse_args(argv)
    raw_root, results_dir, shards_dir, stylegan_xl_repo = resolve_paths(args)
    inventory_path = args.inventory or os.path.join(results_dir, "inventory.json")
    progress_path = args.progress_file or os.path.join(shards_dir, "progress.json")

    print(f"[extract_shards] raw_root={raw_root}")
    print(f"[extract_shards] shards_dir={shards_dir}")
    print(f"[extract_shards] inventory={inventory_path}")
    print(f"[extract_shards] progress_file={progress_path}")
    print(f"[extract_shards] dry_run={args.dry_run}")

    inventory = load_inventory(inventory_path)
    site_entries = get_site_entries(inventory)
    progress = load_progress(progress_path)

    done_count = sum(1 for v in progress.values() if v.get("status") == "done")
    print(f"[extract_shards] {len(site_entries)} site kaydı, {done_count} tanesi progress.json'a göre zaten tamam.")

    if args.dry_run:
        for site in site_entries:
            key = f"round_{site['round']}_site_{site['site']}"
            status = progress.get(key, {}).get("status", "bekliyor")
            note = ""
            if "error" in site:
                note = f" (inventory hatası: {site['error']}, işlenemez)"
            print(f"  {key}: {status}{note}")
        return 0

    os.makedirs(shards_dir, exist_ok=True)

    for site in site_entries:
        key = f"round_{site['round']}_site_{site['site']}"
        if progress.get(key, {}).get("status") == "done":
            print(f"[extract_shards] {key} zaten tamam, atlanıyor.")
            continue

        print(f"\n=== {key} ===")
        prefixes = resolve_prefixes(site)
        pkl_path = site.get("pkl_path")
        if not pkl_path:
            raise RuntimeError(f"{key}: inventory.json'da 'pkl_path' yok.")

        data = load_network_pkl(pkl_path, stylegan_xl_repo)
        g_ema = data["G_ema"]
        full_state_dict = {k: v.detach().cpu().to(torch.float32) for k, v in g_ema.state_dict().items()}

        shard = extract_prefixed_state_dict(full_state_dict, prefixes)
        print(f"[extract_shards] {key}: {len(shard)} tensör, prefixler: {prefixes}")

        round_dir = os.path.join(shards_dir, f"round_{site['round']}")
        os.makedirs(round_dir, exist_ok=True)
        shard_path = os.path.join(round_dir, f"site_{site['site']}.pt")
        torch.save(shard, shard_path)

        meta = build_shard_meta(
            full_state_dict,
            shard,
            extra={
                "round": site["round"],
                "site": site["site"],
                "source_pkl": pkl_path,
                "prefixes_used": prefixes,
            },
        )
        meta_path = os.path.join(round_dir, f"site_{site['site']}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str, ensure_ascii=False)

        print(f"[extract_shards] Yazıldı: {shard_path}")
        print(f"[extract_shards] Yazıldı: {meta_path}")

        progress[key] = {"status": "done", "shard_path": shard_path, "meta_path": meta_path}
        save_progress(progress_path, progress)

        del data, g_ema, full_state_dict, shard
        gc.collect()

    print(f"\n[extract_shards] Bitti. progress.json: {progress_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
