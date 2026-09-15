"""circuits/export_mapping.py testleri.

`onnx`/`onnxruntime` StyleGAN-XL/Drive/ezkl gibi Colab'a özgü DEĞİL —
sade CPU paketleri. Bu yüzden Faz B/C0/C1'in "BİLİNÇLİ TEST SINIRI"
deseninin aksine, export + onnxruntime karşılaştırmasının TAMAMI
sentetik ağırlıklarla burada GERÇEKTEN test edilir (kurulu değilse
zarifçe skip eder — CLAUDE.md madde 7/8 ile tutarlı).
"""

import json
import os

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from circuits.export_mapping import (  # noqa: E402
    compare_onnx_vs_reference,
    compare_onnx_vs_torch,
    export_to_onnx,
    main,
    parse_k_values,
    summarize_onnx_graph,
)
from circuits.rebuild_mapping import build_mapping_network_from_shard, build_pruned_mapping_network  # noqa: E402

Z_DIM = 4
EMBED_NUM = 10
EMBED_DIM = 6
W_DIM = 5
C_DIM = 3


def _make_fake_shard(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "mapping.embed.weight": torch.randn(EMBED_NUM, EMBED_DIM, generator=g),
        "mapping.embed_proj.weight": torch.randn(Z_DIM, EMBED_DIM, generator=g),
        "mapping.embed_proj.bias": torch.randn(Z_DIM, generator=g),
        "mapping.fc0.weight": torch.randn(W_DIM, 2 * Z_DIM, generator=g),
        "mapping.fc0.bias": torch.randn(W_DIM, generator=g),
        "mapping.fc1.weight": torch.randn(W_DIM, W_DIM, generator=g),
        "mapping.fc1.bias": torch.randn(W_DIM, generator=g),
    }


def _make_example_inputs(k: int = 4, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    z = torch.randn(k, Z_DIM)
    idx = torch.randint(0, C_DIM, (k,))
    c = F.one_hot(idx, num_classes=C_DIM).to(torch.float32)
    return z, c


def _export_fake_model(tmp_path, embed_mode: str = "gather"):
    shard = _make_fake_shard()
    model = build_pruned_mapping_network(shard, num_classes=C_DIM, embed_mode=embed_mode)
    model.eval()
    z, c = _make_example_inputs()
    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(model, z, c, str(onnx_path), opset_version=13)
    return model, z, c, str(onnx_path)


def test_export_to_onnx_writes_nonempty_file(tmp_path):
    _model, _z, _c, onnx_path = _export_fake_model(tmp_path)
    assert os.path.getsize(onnx_path) > 0


def test_summarize_onnx_graph_reports_nodes(tmp_path):
    _model, _z, _c, onnx_path = _export_fake_model(tmp_path)
    summary = summarize_onnx_graph(onnx_path)
    assert summary["num_nodes"] > 0
    assert sum(summary["op_counts"].values()) == summary["num_nodes"]
    print(f"ONNX op dağılımı (sentetik model): {summary['op_counts']}")


def test_compare_onnx_vs_torch_matches_within_tolerance(tmp_path):
    model, z, c, onnx_path = _export_fake_model(tmp_path)
    stats = compare_onnx_vs_torch(onnx_path, model, z, c)
    assert stats["max_abs_diff"] < 1e-4


def test_compare_onnx_vs_reference_matches_within_tolerance(tmp_path):
    model, z, c, onnx_path = _export_fake_model(tmp_path)
    with torch.no_grad():
        w = model(z, c)
    reference = {"z": z, "c": c, "w": w.unsqueeze(1).repeat(1, 16, 1)}

    stats = compare_onnx_vs_reference(onnx_path, reference)
    assert stats["max_abs_diff"] < 1e-4


def test_compare_onnx_vs_reference_raises_when_num_ws_copies_differ(tmp_path):
    _model, z, c, onnx_path = _export_fake_model(tmp_path)
    reference = {"z": z, "c": c, "w": torch.randn(z.shape[0], 16, W_DIM)}
    with pytest.raises(RuntimeError, match="BİREBİR AYNI değil"):
        compare_onnx_vs_reference(onnx_path, reference)


def test_parse_k_values_parses_comma_separated_ints():
    assert parse_k_values("1,4,8") == [1, 4, 8]


def test_parse_k_values_raises_on_empty():
    with pytest.raises(ValueError):
        parse_k_values("")


def test_gather_mode_onnx_graph_contains_argmax_and_gather(tmp_path):
    _model, _z, _c, onnx_path = _export_fake_model(tmp_path, embed_mode="gather")
    summary = summarize_onnx_graph(onnx_path)
    assert "ArgMax" in summary["op_counts"]
    assert "Gather" in summary["op_counts"]


def test_matmul_mode_onnx_graph_has_no_argmax_or_gather(tmp_path):
    _model, _z, _c, onnx_path = _export_fake_model(tmp_path, embed_mode="matmul")
    summary = summarize_onnx_graph(onnx_path)
    assert "ArgMax" not in summary["op_counts"]
    assert "Gather" not in summary["op_counts"]
    # matmul modu ek bir MatMul uretir (embed secimi icin) - toplam MatMul
    # sayisi gather moduna gore en az 1 fazla olmali.
    gather_summary = summarize_onnx_graph(_export_fake_model(tmp_path, embed_mode="gather")[3])
    assert summary["op_counts"].get("MatMul", 0) > gather_summary["op_counts"].get("MatMul", 0)


def _write_shard_and_reference(tmp_path, k: int = 4, seed_shard: int = 123, seed_z: int = 7) -> tuple[str, str]:
    shard = _make_fake_shard(seed=seed_shard)
    round_dir = tmp_path / "shards" / "round_14"
    round_dir.mkdir(parents=True)
    torch.save(shard, round_dir / "site_0.pt")

    full_model = build_mapping_network_from_shard(shard)
    full_model.eval()
    torch.manual_seed(seed_z)
    z = torch.randn(k, Z_DIM)
    idx = torch.tensor([i % C_DIM for i in range(k)])
    c = F.one_hot(idx, num_classes=C_DIM).to(torch.float32)
    with torch.no_grad():
        w = full_model(z, c)
    w_full = w.unsqueeze(1).repeat(1, 16, 1)

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    torch.save(
        {"z": z, "c": c, "w": w_full, "seed": seed_z, "source_pkl": "fake.pkl"},
        reference_dir / f"mapping_ref_k{k}.pt",
    )

    return str(tmp_path / "shards"), str(reference_dir)


def test_main_exports_both_variants_plus_legacy_copy_and_json(tmp_path):
    shards_dir, reference_dir = _write_shard_and_reference(tmp_path, k=4)
    onnx_dir = tmp_path / "onnx"

    exit_code = main(
        [
            "--shards-dir", shards_dir,
            "--reference-dir", reference_dir,
            "--onnx-dir", str(onnx_dir),
            "--round", "14",
            "--site", "0",
            "--k-values", "4",
        ]
    )

    assert exit_code == 0
    gather_path = onnx_dir / "mapping_k4_gather.onnx"
    matmul_path = onnx_dir / "mapping_k4_matmul.onnx"
    legacy_path = onnx_dir / "mapping_k4.onnx"
    variants_json_path = onnx_dir / "onnx_variants.json"

    assert gather_path.exists()
    assert matmul_path.exists()
    assert legacy_path.exists()
    assert variants_json_path.exists()

    # geriye uyumluluk kopyası == matmul dosyası (byte-eşit)
    assert legacy_path.read_bytes() == matmul_path.read_bytes()

    with open(variants_json_path, encoding="utf-8") as f:
        variants = json.load(f)

    k4 = variants["k4"]
    assert k4["default_onnx_mode"] == "matmul"
    assert "ArgMax" in k4["gather"]["op_counts"]
    assert "ArgMax" not in k4["matmul"]["op_counts"]
    assert k4["pruned_vs_full_max_abs_diff"] == 0.0
    assert k4["embed_mode_comparison_max_abs_diff"] < 1e-6
