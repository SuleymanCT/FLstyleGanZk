#!/usr/bin/env python
"""Faz C1 kabul testi: yeniden kurulmuş mapping ağı, Faz C0'ın gerçek
StyleGAN-XL referans çıktısıyla 1e-5 içinde eşleşiyor mu?

BİLİNÇLİ TEST SINIRI: gerçek bir shard (`scripts/extract_shards.py`
çıktısı) ve gerçek bir referans dosyası (`scripts/make_reference_outputs.py`
çıktısı) gerektirir — ikisi de Drive'da, yerelde yok. Bu yüzden
`main()`'in birim testi yoktur. `circuits/rebuild_mapping.py` ve
`circuits/compare_utils.py`'nin saf mantığı ayrı test edilmiştir.

Kullanım (Colab'da):
    python -m scripts.verify_rebuild --env colab --k 4
"""

from __future__ import annotations

import argparse
import os

import torch

from circuits.compare_utils import compute_comparison_stats
from circuits.rebuild_mapping import build_mapping_network_from_shard, normalize_2nd_moment
from configs.loader import load_paths
from storage.hashing import canonical_hash


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Faz C1: rebuild_mapping'i gerçek referansla karşılaştır.")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--shards-dir", default=None)
    parser.add_argument("--reference-dir", default=None, help="varsayılan: {zk_root}/reference")
    parser.add_argument("--round", type=int, default=14)
    parser.add_argument("--site", type=int, default=0)
    parser.add_argument("--k", type=int, default=4, help="karşılaştırılacak referans dosyası: mapping_ref_k{k}.pt")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args(argv)


def resolve_paths(args) -> tuple[str, str]:
    paths = load_paths(env=args.env, config_path=args.paths_config)
    shards_dir = args.shards_dir or paths["shards_dir"]
    reference_dir = args.reference_dir or os.path.join(paths["zk_root"], "reference")
    return shards_dir, reference_dir


def load_shard(shards_dir: str, round_idx: int, site_idx: int) -> dict:
    shard_path = os.path.join(shards_dir, f"round_{round_idx}", f"site_{site_idx}.pt")
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(
            f"Shard bulunamadı: '{shard_path}'. Önce 'python -m scripts.extract_shards' çalışmış olmalı."
        )
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    return {k: v.detach().to(torch.float32) for k, v in shard.items()}


def load_reference(reference_dir: str, k: int) -> dict:
    ref_path = os.path.join(reference_dir, f"mapping_ref_k{k}.pt")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(
            f"Referans dosyası bulunamadı: '{ref_path}'. Önce 'python -m scripts.make_reference_outputs' çalışmış olmalı."
        )
    return torch.load(ref_path, map_location="cpu", weights_only=False)


def assert_num_ws_copies_identical(w: torch.Tensor) -> None:
    """Referanstaki w'nin (k, num_ws, w_dim) şeklinde num_ws boyunca
    HEPSİ birebir aynı olmalı (broadcast'ten geldiği için). Değilse
    referans dosyası beklenenden farklı — net hata."""
    if w.dim() != 3:
        raise RuntimeError(f"Referans 'w' 3 boyutlu (k, num_ws, w_dim) bekleniyordu, bulunan şekil: {tuple(w.shape)}")
    reference_slice = w[:, 0:1, :]
    if not torch.allclose(w, reference_slice.expand_as(w)):
        max_diff = (w - reference_slice.expand_as(w)).abs().max().item()
        raise RuntimeError(
            f"Referanstaki w'nin num_ws kopyaları BİREBİR AYNI değil (max fark={max_diff}). "
            f"Bu beklenmiyordu (broadcast salt tekrar olmalıydı) — referans dosyasını/varsayımları gözden geçir."
        )


def print_intermediate_diagnostics(model, z: torch.Tensor, c: torch.Tensor, reference: dict) -> None:
    with torch.no_grad():
        x0 = normalize_2nd_moment(z.to(torch.float32))
        indices = c.argmax(dim=1)
        embed_out = model.embed(indices)
        embed_proj_out = model.embed_proj(embed_out)
        y = normalize_2nd_moment(embed_proj_out)
        cat = torch.cat([x0, y], dim=1)
        fc0_out = model.fc0(cat)
        fc1_out = model.fc1(fc0_out)

    rebuilt_intermediates = {"embed_proj_out": embed_proj_out, "fc0_out": fc0_out, "fc1_out": fc1_out}
    reference_intermediates = reference.get("intermediates")

    if not reference_intermediates:
        print(
            "[verify_rebuild] Referans dosyasında 'intermediates' yok (eski format) — "
            "katman-bazlı karşılaştırma yapılamıyor. Sadece rebuild'in kendi ara değer istatistikleri:"
        )
        for name, tensor in rebuilt_intermediates.items():
            print(f"  {name}: norm={tensor.norm().item():.6f} mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        print(
            "[verify_rebuild] Tam teşhis için 'python -m scripts.make_reference_outputs' yeniden çalıştırılabilir "
            "(artık ara çıktıları da kaydediyor)."
        )
        return

    print("[verify_rebuild] Katman katman karşılaştırma (referansta 'intermediates' bulundu):")
    first_divergence = None
    for name, rebuilt_tensor in rebuilt_intermediates.items():
        ref_tensor = reference_intermediates.get(name)
        if ref_tensor is None:
            print(f"  {name}: referansta yok, atlanıyor.")
            continue
        stats = compute_comparison_stats(rebuilt_tensor, ref_tensor)
        print(f"  {name}: max_abs_diff={stats['max_abs_diff']:.8f} cosine={stats['cosine_similarity']}")
        if first_divergence is None and stats["max_abs_diff"] > 1e-5:
            first_divergence = name

    if first_divergence:
        print(f"[verify_rebuild] İLK ayrışan katman: '{first_divergence}'")


def main(argv=None) -> int:
    args = parse_args(argv)
    shards_dir, reference_dir = resolve_paths(args)

    print(f"[verify_rebuild] shards_dir={shards_dir}")
    print(f"[verify_rebuild] reference_dir={reference_dir}")
    print(f"[verify_rebuild] round={args.round} site={args.site} k={args.k} tolerance={args.tolerance}")

    shard = load_shard(shards_dir, args.round, args.site)
    reference = load_reference(reference_dir, args.k)

    shard_hash = canonical_hash(shard)
    ref_shard_hash = reference.get("shard_hash")
    print(f"[verify_rebuild] shard_hash (hesaplanan)={shard_hash}")
    print(f"[verify_rebuild] shard_hash (referansta)={ref_shard_hash}")
    if shard_hash != ref_shard_hash:
        raise RuntimeError(
            f"Shard ve referans dosyası AYNI ağırlıklardan gelmiyor "
            f"(shard_hash uyuşmuyor: {shard_hash} != {ref_shard_hash}). "
            f"Yanlış (round,site)/k çiftini karşılaştırıyor olabilirsin."
        )
    print("[verify_rebuild] shard_hash eşleşti — doğru ağırlık/referans çifti.")

    model = build_mapping_network_from_shard(shard)
    model.eval()

    w_ref_full = reference["w"]
    assert_num_ws_copies_identical(w_ref_full)
    w_ref = w_ref_full[:, 0, :]

    z, c = reference["z"], reference["c"]
    with torch.no_grad():
        w_rebuilt = model(z, c)

    print(f"[verify_rebuild] w_ref şekli: {tuple(w_ref.shape)}, w_rebuilt şekli: {tuple(w_rebuilt.shape)}")

    stats = compute_comparison_stats(w_rebuilt, w_ref)
    print(f"[verify_rebuild] max_abs_diff={stats['max_abs_diff']}")
    print(f"[verify_rebuild] mean_abs_diff={stats['mean_abs_diff']}")
    print(f"[verify_rebuild] cosine_similarity={stats['cosine_similarity']}")

    if stats["max_abs_diff"] > args.tolerance:
        print(f"\n[verify_rebuild] EŞİK AŞILDI ({stats['max_abs_diff']} > {args.tolerance}) — teşhis:")
        print_intermediate_diagnostics(model, z, c, reference)
        raise RuntimeError(
            f"max_abs_diff ({stats['max_abs_diff']}) tolerance'ı ({args.tolerance}) aşıyor. "
            f"rebuild_mapping.py'nin matematiği/mimari sabitleri gözden geçirilmeli."
        )

    print(f"\n[verify_rebuild] BAŞARILI: max_abs_diff {args.tolerance} eşiğinin altında.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
