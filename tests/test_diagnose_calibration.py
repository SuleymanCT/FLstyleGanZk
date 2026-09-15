"""scripts/diagnose_calibration.py testleri.

ezkl GEREKTİRMEYEN saf yardımcı fonksiyonlar (tensör istatistikleri,
input.json şekli ayrıştırma, ONNX girdi şekli okuma, çapraz kontrol,
max_logrows ayrıştırma, tablo yazdırma) burada GERÇEKTEN test edilir.
Modül kendi top-level'ında `import ezkl` YAPMAZ (sadece `dump_full_settings`/
`attempt_calibration`/`attempt_skip_calibration` içinde, çağrıldıklarında),
bu yüzden bu dosya yerelde (ezkl kurulu değilken) sorunsuz çalışır.
"""

import torch

from scripts.diagnose_calibration import (
    cross_check_input_shapes,
    describe_input_json_shape,
    describe_tensor_stats,
    parse_max_logrows_values,
    print_calibration_table,
    print_skip_calibration_table,
    read_onnx_input_shapes,
)


def test_describe_tensor_stats_reports_correct_values():
    t = torch.tensor([[-2.0, 1.0], [3.0, 0.0]])
    stats = describe_tensor_stats(t)
    assert stats["shape"] == [2, 2]
    assert stats["min"] == -2.0
    assert stats["max"] == 3.0
    assert stats["mean"] == 0.5
    assert stats["abs_max"] == 3.0


def test_describe_input_json_shape_counts_inputs_and_lengths():
    payload = {"input_data": [[1.0, 2.0, 3.0], [0.0, 1.0]]}
    assert describe_input_json_shape(payload) == {"num_inputs": 2, "lengths": [3, 2]}


def test_describe_input_json_shape_handles_missing_key():
    assert describe_input_json_shape({}) == {"num_inputs": 0, "lengths": []}


def test_read_onnx_input_shapes_matches_export(tmp_path):
    class TwoInputModel(torch.nn.Module):
        def forward(self, z, c):
            return z.sum(dim=1, keepdim=True) + c.sum(dim=1, keepdim=True)

    model = TwoInputModel()
    z = torch.randn(4, 64)
    c = torch.randn(4, 5)
    onnx_path = tmp_path / "two_input.onnx"
    torch.onnx.export(model, (z, c), str(onnx_path), dynamo=False, input_names=["z", "c"], output_names=["w"])

    shapes = read_onnx_input_shapes(str(onnx_path))
    assert shapes["z"] == [4, 64]
    assert shapes["c"] == [4, 5]


def test_cross_check_input_shapes_detects_match():
    onnx_shapes = {"z": [4, 64], "c": [4, 5]}
    assert cross_check_input_shapes(onnx_shapes, [256, 20]) == []


def test_cross_check_input_shapes_detects_length_mismatch():
    onnx_shapes = {"z": [4, 64], "c": [4, 5]}
    warnings = cross_check_input_shapes(onnx_shapes, [256, 999])
    assert len(warnings) == 1
    assert "'c'" in warnings[0]


def test_cross_check_input_shapes_detects_input_count_mismatch():
    onnx_shapes = {"z": [4, 64], "c": [4, 5]}
    warnings = cross_check_input_shapes(onnx_shapes, [256])
    assert len(warnings) == 1
    assert "girdi sayısı" in warnings[0]


def test_parse_max_logrows_values_parses_none_and_ints():
    assert parse_max_logrows_values("none,15,19,22") == [None, 15, 19, 22]


def test_parse_max_logrows_values_handles_empty_tokens():
    assert parse_max_logrows_values("15,,22,") == [15, 22]


def test_print_calibration_table_smoke(capsys):
    results = [
        {"method": "run_args_scale", "target": "resources", "scale": 8, "max_logrows": None, "status": "failed", "elapsed_seconds": 1.23, "error": "boom"},
        {"method": "scales_kwarg", "target": "resources", "scale": 13, "max_logrows": 19, "status": "success", "elapsed_seconds": 4.56, "error": None},
    ]
    print_calibration_table(results)
    out = capsys.readouterr().out
    assert "run_args_scale" in out and "scales_kwarg" in out
    assert "boom" in out


def test_print_skip_calibration_table_smoke(capsys):
    results = [{"scale": 13, "status": "success", "last_step": "prove", "error": None}]
    print_skip_calibration_table(results)
    out = capsys.readouterr().out
    assert "prove" in out and "success" in out
