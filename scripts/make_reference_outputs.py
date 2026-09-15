#!/usr/bin/env python
"""Faz C0: gerçek StyleGAN-XL mapping ağından referans (z,c)->w çıktıları üret.

Faz C1 (StyleGAN-XL'e bağımlı OLMAYAN, yeniden kurulmuş minimal mapping
modülü) burada üretilen dosyalarla 1e-5 içinde eşleşmeli mi diye
karşılaştırılacak.

BİLİNÇLİ TEST SINIRI: `--dry-run` dışındaki akış gerçek bir
network-snapshot.pkl ve gerçek bir StyleGAN-XL klonu gerektirir.
Yerelde ikisi de yok; bu yüzden `main()`'in birim testi yoktur.
`resolve_site_pkl_path` (dizin taraması, dosya açmaz) ve saf mantık
(`fl/reference_utils.py`) ayrı test edilmiştir.

Kullanım (Colab'da):
    python -m scripts.make_reference_outputs --env colab --dry-run
    python -m scripts.make_reference_outputs --env colab
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os

import torch

from configs.loader import load_paths
from fl.module_tree import detect_candidate_submodules, extract_known_attrs
from fl.reference_utils import build_z_c, resolve_mapping_kwargs
from fl.shard_utils import extract_prefixed_state_dict
from fl.stylegan_xl_env import load_network_pkl
from scripts.inventory import find_run_subdir, find_snapshot_pkl
from storage.hashing import canonical_hash
from storage.pathguard import assert_writable

DEFAULT_K_VALUES = "1,4,8"
DEFAULT_TRUNCATION_PSI = 1.0
DEFAULT_TRUNCATION_CUTOFF = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Faz C0: gerçek mapping ağından referans çıktıları üret.")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--reference-dir", default=None, help="varsayılan: {zk_root}/reference")
    parser.add_argument("--stylegan-xl-repo", default=None)
    parser.add_argument("--round", type=int, default=14, help="varsayılan: round_14")
    parser.add_argument("--site", type=int, default=0, help="varsayılan: site_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-values", default=DEFAULT_K_VALUES, help="virgülle ayrılmış k listesi, ör. '1,4,8'")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="pkl açmadan sadece hangi dosyanın bulunacağını/üretileceğini raporla.",
    )
    return parser.parse_args(argv)


def resolve_paths(args) -> tuple[str, str, str]:
    paths = load_paths(env=args.env, config_path=args.paths_config)
    raw_root = args.raw_root or paths["raw_root"]
    reference_dir = args.reference_dir or os.path.join(paths["zk_root"], "reference")
    stylegan_xl_repo = args.stylegan_xl_repo or paths["stylegan_xl_repo"]
    return raw_root, reference_dir, stylegan_xl_repo


def parse_k_values(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError(f"--k-values boş/geçersiz: {raw!r}")
    return values


def resolve_site_pkl_path(raw_root: str, round_idx: int, site_idx: int) -> str:
    """`scripts.inventory`'nin zaten test edilmiş dizin-tarama
    fonksiyonlarını yeniden kullanır — inventory.json'a bağımlı DEĞİL,
    bu scripti self-contained yapar (Faz A1'in ayrıca koşmuş olması
    gerekmez). Dosya açmaz, sadece glob yapar.
    """
    site_dir = os.path.join(raw_root, f"round_{round_idx}", f"site_{site_idx}")
    if not os.path.isdir(site_dir):
        raise FileNotFoundError(f"Site dizini bulunamadı: '{site_dir}'.")

    run_dir, run_dir_warning = find_run_subdir(site_dir)
    if run_dir is None:
        raise RuntimeError(f"round {round_idx} site {site_idx}: {run_dir_warning}")

    pkl_path, pkl_warning = find_snapshot_pkl(run_dir)
    if pkl_warning:
        print(f"UYARI: round {round_idx} site {site_idx}: {pkl_warning}")
    if pkl_path is None:
        raise FileNotFoundError(f"round {round_idx} site {site_idx}: '{run_dir}' altında pkl bulunamadı.")

    return pkl_path


def main(argv=None) -> int:
    args = parse_args(argv)
    raw_root, reference_dir, stylegan_xl_repo = resolve_paths(args)
    k_values = parse_k_values(args.k_values)

    print(f"[make_reference_outputs] raw_root={raw_root}")
    print(f"[make_reference_outputs] reference_dir={reference_dir}")
    print(f"[make_reference_outputs] round={args.round} site={args.site}")
    print(f"[make_reference_outputs] seed={args.seed} k_values={k_values}")
    print(f"[make_reference_outputs] dry_run={args.dry_run}")

    pkl_path = resolve_site_pkl_path(raw_root, args.round, args.site)
    print(f"[make_reference_outputs] pkl bulundu: {pkl_path}")

    if args.dry_run:
        print(f"[make_reference_outputs] --dry-run: pkl açılmadı. Üretilecek dosyalar (henüz yazılmadı):")
        for k in k_values:
            print(f"  {os.path.join(reference_dir, f'mapping_ref_k{k}.pt')}")
        return 0

    data = load_network_pkl(pkl_path, stylegan_xl_repo)
    if "G_ema" not in data:
        raise RuntimeError(f"pkl içinde 'G_ema' anahtarı yok. Bulunan anahtarlar: {sorted(data.keys())}")
    g_ema = data["G_ema"]

    dims = extract_known_attrs(g_ema, names=("z_dim", "c_dim", "w_dim", "num_ws"))
    print(f"[make_reference_outputs] Boyutlar — bulunan: {dims['found']}, bulunamayan: {dims['bulunamadi']}")
    if dims["bulunamadi"]:
        raise RuntimeError(f"G_ema'da beklenen boyut attribute'ları eksik: {dims['bulunamadi']}")
    z_dim = dims["found"]["z_dim"]
    c_dim = dims["found"]["c_dim"]

    candidates = detect_candidate_submodules(g_ema)
    prefixes = sorted(set(candidates.get("mapping", [])) | set(candidates.get("embedding", [])))
    print(f"[make_reference_outputs] Mapping/embedding adayları: {candidates}")
    if not prefixes:
        raise RuntimeError(
            "mapping/embedding alt modülü tespit edilemedi — referans üretilemiyor "
            "(fl.module_tree.DEFAULT_KEYWORD_GROUPS'u genişletmek gerekebilir)."
        )
    print(f"[make_reference_outputs] shard prefixleri: {prefixes}")

    full_state_dict = {k: v.detach().cpu().to(torch.float32) for k, v in g_ema.state_dict().items()}
    shard = extract_prefixed_state_dict(full_state_dict, prefixes)
    shard_hash = canonical_hash(shard)
    print(f"[make_reference_outputs] shard_hash={shard_hash}")

    mapping = g_ema.mapping
    signature = inspect.signature(mapping.forward)
    print(f"[make_reference_outputs] G_ema.mapping.forward imzası: {signature}")
    print(f"[make_reference_outputs] Parametreler: {list(signature.parameters.keys())}")

    assert_writable(reference_dir)
    os.makedirs(reference_dir, exist_ok=True)

    for k in k_values:
        print(f"\n=== k={k} ===")
        z, c = build_z_c(k, z_dim, c_dim, args.seed)
        extra_kwargs = resolve_mapping_kwargs(signature, DEFAULT_TRUNCATION_PSI, DEFAULT_TRUNCATION_CUTOFF)
        print(f"[make_reference_outputs] mapping çağrı kwargs: {extra_kwargs}")

        with torch.no_grad():
            w = mapping(z, c, **extra_kwargs)

        print(f"[make_reference_outputs] w şekli: {tuple(w.shape)} dtype={w.dtype}")

        out_path = os.path.join(reference_dir, f"mapping_ref_k{k}.pt")
        assert_writable(out_path)
        torch.save(
            {
                "z": z,
                "c": c,
                "w": w.detach().cpu(),
                "seed": args.seed,
                "source_pkl": pkl_path,
                "shard_hash": shard_hash,
            },
            out_path,
        )
        print(f"[make_reference_outputs] Yazıldı: {out_path}")

    del data, g_ema, full_state_dict, shard
    gc.collect()

    print("\n[make_reference_outputs] Bitti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
