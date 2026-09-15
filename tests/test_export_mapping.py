"""circuits/export_mapping.py testleri.

`onnx`/`onnxruntime` StyleGAN-XL/Drive/ezkl gibi Colab'a özgü DEĞİL —
sade CPU paketleri. Bu yüzden Faz B/C0/C1'in "BİLİNÇLİ TEST SINIRI"
deseninin aksine, export + onnxruntime karşılaştırmasının TAMAMI
sentetik ağırlıklarla burada GERÇEKTEN test edilir (kurulu değilse
zarifçe skip eder — CLAUDE.md madde 7/8 ile tutarlı).
"""

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
    parse_k_values,
    summarize_onnx_graph,
)
from circuits.rebuild_mapping import build_pruned_mapping_network  # noqa: E402

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


def _export_fake_model(tmp_path):
    shard = _make_fake_shard()
    model = build_pruned_mapping_network(shard, num_classes=C_DIM)
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
