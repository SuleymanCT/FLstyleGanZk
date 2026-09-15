#!/usr/bin/env python
"""Faz C2: budanmış mapping modülünü ONNX'e ihraç et.

`embed` tablosu (1000×320, StyleGAN-XL'in ImageNet varsayılanından
kalma) bizim görevde sadece `c_dim` (5) satır kullanıyor — 995 satır
ölü ağırlık. Bu modül önce budamanın GERÇEK verilerle güvenli olduğunu
doğrular (`circuits.rebuild_mapping.verify_prunable_embed_rows`),
budanmış modülün çıktısının budanmamışla birebir aynı olduğunu
kanıtlar, sonra ONNX'e ihraç edip onnxruntime ile hem kendi çıktısına
hem Faz C0'ın gerçek referansına karşı doğrular.

BİLİNÇLİ TEST SINIRI (kısmi): `export_to_onnx`/`summarize_onnx_graph`/
`compare_onnx_vs_torch`/`compare_onnx_vs_reference` — StyleGAN-XL/Drive
GEREKTİRMEZ (sadece `onnx`+`onnxruntime`, sade CPU paketleri), bu
yüzden `tests/test_export_mapping.py`'de sentetik ağırlıklarla
GERÇEKTEN test edilir. Sadece `main()`'in gerçek shard+referans
gerektiren kısmı Colab'a kalır.

Kullanım (Colab'da):
    python -m circuits.export_mapping --env colab --k-values 1,4,8
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import onnx
import onnxruntime as ort
import torch

from circuits.compare_utils import assert_num_ws_copies_identical, compute_comparison_stats
from circuits.rebuild_mapping import build_mapping_network_from_shard, build_pruned_mapping_network, verify_prunable_embed_rows
from configs.loader import load_paths
from storage.pathguard import assert_writable

DEFAULT_OPSET_VERSION = 13
ONNX_VS_TORCH_TOLERANCE = 1e-4


def export_to_onnx(model: torch.nn.Module, example_z: torch.Tensor, example_c: torch.Tensor, onnx_path: str, opset_version: int) -> None:
    """`(z, c) -> w` modülünü ONNX'e ihraç eder. `dynamic_axes` YOK —
    ezkl sabit şekil istiyor, `k` bu ihraçta dondurulan değere sabitlenir.
    """
    model.eval()
    torch.onnx.export(
        model,
        (example_z, example_c),
        str(onnx_path),
        dynamo=False,
        export_params=True,
        opset_version=opset_version,
        input_names=["z", "c"],
        output_names=["w"],
    )
    if not os.path.isfile(onnx_path):
        raise RuntimeError(f"torch.onnx.export sonrası dosya yok: {onnx_path}")


def summarize_onnx_graph(onnx_path: str) -> dict:
    """Grafikteki düğüm sayısını ve operatör dağılımını döner — ezkl'nin
    desteklemeyebileceği bir op (ör. ArgMax/Gather) varsa burada görünür.
    """
    model = onnx.load(str(onnx_path))
    op_counts = Counter(node.op_type for node in model.graph.node)
    return {"num_nodes": len(model.graph.node), "op_counts": dict(op_counts)}


def _run_onnxruntime(onnx_path: str, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {"z": z.detach().cpu().numpy(), "c": c.detach().cpu().numpy()}
    (output,) = session.run(["w"], inputs)
    return torch.from_numpy(output)


def compare_onnx_vs_torch(onnx_path: str, model: torch.nn.Module, z: torch.Tensor, c: torch.Tensor) -> dict:
    with torch.no_grad():
        torch_output = model(z, c)
    onnx_output = _run_onnxruntime(onnx_path, z, c)

    stats = compute_comparison_stats(onnx_output, torch_output)
    if stats["max_abs_diff"] > ONNX_VS_TORCH_TOLERANCE:
        raise RuntimeError(
            f"onnxruntime vs torch max_abs_diff ({stats['max_abs_diff']}) "
            f"{ONNX_VS_TORCH_TOLERANCE} eşiğini aşıyor."
        )
    return stats


def compare_onnx_vs_reference(onnx_path: str, reference: dict) -> dict:
    w_ref_full = reference["w"]
    assert_num_ws_copies_identical(w_ref_full)
    w_ref = w_ref_full[:, 0, :]

    onnx_output = _run_onnxruntime(onnx_path, reference["z"], reference["c"])

    stats = compute_comparison_stats(onnx_output, w_ref)
    if stats["max_abs_diff"] > ONNX_VS_TORCH_TOLERANCE:
        raise RuntimeError(
            f"onnxruntime vs Faz C0 referansı max_abs_diff ({stats['max_abs_diff']}) "
            f"{ONNX_VS_TORCH_TOLERANCE} eşiğini aşıyor."
        )
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Faz C2: budanmış mapping modülünü ONNX'e ihraç et.")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--shards-dir", default=None)
    parser.add_argument("--reference-dir", default=None, help="varsayılan: {zk_root}/reference")
    parser.add_argument("--onnx-dir", default=None, help="varsayılan: {zk_root}/circuits")
    parser.add_argument("--round", type=int, default=14)
    parser.add_argument("--site", type=int, default=0)
    parser.add_argument("--k-values", default="1,4,8")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET_VERSION)
    return parser.parse_args(argv)


def resolve_paths(args) -> tuple[str, str, str]:
    paths = load_paths(env=args.env, config_path=args.paths_config)
    shards_dir = args.shards_dir or paths["shards_dir"]
    reference_dir = args.reference_dir or os.path.join(paths["zk_root"], "reference")
    onnx_dir = args.onnx_dir or os.path.join(paths["zk_root"], "circuits")
    return shards_dir, reference_dir, onnx_dir


def parse_k_values(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError(f"--k-values boş/geçersiz: {raw!r}")
    return values


def load_shard(shards_dir: str, round_idx: int, site_idx: int) -> dict:
    shard_path = os.path.join(shards_dir, f"round_{round_idx}", f"site_{site_idx}.pt")
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(f"Shard bulunamadı: '{shard_path}'.")
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    return {k: v.detach().to(torch.float32) for k, v in shard.items()}


def load_reference(reference_dir: str, k: int) -> dict:
    ref_path = os.path.join(reference_dir, f"mapping_ref_k{k}.pt")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"Referans dosyası bulunamadı: '{ref_path}'.")
    return torch.load(ref_path, map_location="cpu", weights_only=False)


def main(argv=None) -> int:
    args = parse_args(argv)
    shards_dir, reference_dir, onnx_dir = resolve_paths(args)
    k_values = parse_k_values(args.k_values)

    print(f"[export_mapping] shards_dir={shards_dir}")
    print(f"[export_mapping] reference_dir={reference_dir}")
    print(f"[export_mapping] onnx_dir={onnx_dir}")
    print(f"[export_mapping] round={args.round} site={args.site} k_values={k_values} opset={args.opset}")

    shard = load_shard(shards_dir, args.round, args.site)

    assert_writable(onnx_dir)
    os.makedirs(onnx_dir, exist_ok=True)

    for k in k_values:
        print(f"\n=== k={k} ===")
        reference = load_reference(reference_dir, k)
        z, c = reference["z"], reference["c"]
        c_dim = c.shape[1]
        print(f"[export_mapping] Ölçülen c_dim (referanstan): {c_dim}")

        used_indices = verify_prunable_embed_rows([c], num_classes=c_dim)
        print(f"[export_mapping] Gerçek c'de gözlenen sınıf indeksleri: {sorted(used_indices)} (aralık: [0,{c_dim}))")

        full_model = build_mapping_network_from_shard(shard)
        full_model.eval()
        pruned_model = build_pruned_mapping_network(shard, num_classes=c_dim)
        pruned_model.eval()

        full_params = sum(p.numel() for p in full_model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        print(
            f"[export_mapping] Parametre sayısı: tam={full_params}, budanmış={pruned_params} "
            f"(%{100 * (full_params - pruned_params) / full_params:.1f} azalma)"
        )

        with torch.no_grad():
            full_output = full_model(z, c)
            pruned_output = pruned_model(z, c)
        prune_stats = compute_comparison_stats(pruned_output, full_output)
        print(f"[export_mapping] Budanmış vs tam çıktı max_abs_diff: {prune_stats['max_abs_diff']}")
        if prune_stats["max_abs_diff"] != 0.0:
            raise RuntimeError(
                f"Budanmış modülün çıktısı tam modülle AYNI DEĞİL (max_abs_diff={prune_stats['max_abs_diff']}). "
                f"Budama GÜVENLİ DEĞİL — ONNX ihracı yapılmıyor."
            )

        onnx_path = os.path.join(onnx_dir, f"mapping_k{k}.onnx")
        assert_writable(onnx_path)
        export_to_onnx(pruned_model, z, c, onnx_path, args.opset)
        print(f"[export_mapping] Yazıldı: {onnx_path}")

        graph_summary = summarize_onnx_graph(onnx_path)
        print(f"[export_mapping] ONNX grafiği: {graph_summary['num_nodes']} düğüm, op dağılımı: {graph_summary['op_counts']}")

        torch_stats = compare_onnx_vs_torch(onnx_path, pruned_model, z, c)
        print(f"[export_mapping] onnxruntime vs torch: max_abs_diff={torch_stats['max_abs_diff']}")

        ref_stats = compare_onnx_vs_reference(onnx_path, reference)
        print(f"[export_mapping] onnxruntime vs Faz C0 referansı: max_abs_diff={ref_stats['max_abs_diff']}")

    print("\n[export_mapping] Bitti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
