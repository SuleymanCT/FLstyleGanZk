"""scripts/bench_circuit.py testleri.

ezkl/anvil/solc/web3 GEREKTİRMEYEN saf yardımcı fonksiyonlar (ızgara
kurma, kombinasyon anahtarlama, log ayrıştırma, devre istatistik
çıkarımı, input.json üretimi, ilerleme dosyası okuma/yazma, markdown
tablo üretimi) burada GERÇEKTEN test edilir — modül `import ezkl`'i
kendi top-level'ında YAPMAZ (sadece `run_ezkl_pipeline`/`run_prove_and_verify`/
`run_worker` içinde, çağrıldıklarında yapar), bu yüzden bu dosya
yerelde (Windows, ezkl/anvil/solc/web3 hiçbiri kurulu değilken) sorunsuz
çalışır.
"""

import json

import pytest
import torch

from scripts.bench_circuit import (
    build_grid,
    build_multi_input_json,
    combo_key,
    count_decomposition_warnings,
    extract_circuit_stats,
    extract_max_abs_error,
    force_settings_scale,
    load_bench_results,
    render_markdown_table,
    save_bench_results,
    write_json_file,
)


def test_build_grid_order_starts_with_smallest_k_first_mode_lowest_scale():
    grid = build_grid([1, 4, 8], ["matmul", "gather"], [8, 11, 13])
    assert grid[0] == {"k": 1, "embed_mode": "matmul", "scale": 8}
    assert len(grid) == 3 * 2 * 3


def test_build_grid_only_first_returns_single_combo():
    grid = build_grid([1, 4, 8], ["matmul", "gather"], [8, 11, 13], only_first=True)
    assert grid == [{"k": 1, "embed_mode": "matmul", "scale": 8}]


def test_build_grid_rejects_invalid_embed_mode():
    with pytest.raises(ValueError, match="embed_mode"):
        build_grid([1], ["bogus"], [8])


def test_combo_key_format():
    assert combo_key({"k": 4, "embed_mode": "gather", "scale": 11}) == "k4_gather_scale11"


def test_count_decomposition_warnings_counts_case_insensitive():
    log = "foo\nDecomposition error: integer 123 is too large\nbar\ndecomposition error: integer 999 is too large\n"
    assert count_decomposition_warnings(log) == 2


def test_count_decomposition_warnings_zero_when_absent():
    assert count_decomposition_warnings("her sey normal, uyari yok") == 0


def test_extract_max_abs_error_finds_value():
    log = "Numerical Fidelity Report: max_abs_error=0.00033\ndone"
    assert extract_max_abs_error(log) == pytest.approx(0.00033)


def test_extract_max_abs_error_returns_none_when_absent():
    assert extract_max_abs_error("hicbir sey yok burada") is None


def test_extract_max_abs_error_returns_last_match():
    log = "max_abs_error=0.1 ... sonra tekrar calisti max_abs_error=0.02"
    assert extract_max_abs_error(log) == pytest.approx(0.02)


def test_extract_circuit_stats_collects_known_keys():
    settings = {
        "run_args": {"logrows": 17, "input_scale": 11, "param_scale": 11, "num_inner_cols": 2},
        "num_rows": 12345,
        "required_lookups": ["a", "b", "c"],
    }
    stats = extract_circuit_stats(settings)
    assert stats["logrows"] == 17
    assert stats["input_scale"] == 11
    assert stats["num_rows"] == 12345
    assert stats["num_required_lookups"] == 3


def test_extract_circuit_stats_handles_missing_keys_gracefully():
    assert extract_circuit_stats({}) == {}
    assert extract_circuit_stats({"run_args": "not_a_dict"}) == {}


def test_force_settings_scale_overwrites_run_args(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"run_args": {"input_scale": 7, "param_scale": 7, "other": "x"}}), encoding="utf-8")

    result = force_settings_scale(str(settings_path), scale=13)

    assert result["run_args"]["input_scale"] == 13
    assert result["run_args"]["param_scale"] == 13
    assert result["run_args"]["other"] == "x"

    with open(settings_path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["run_args"]["input_scale"] == 13


def test_force_settings_scale_warns_and_skips_when_run_args_missing(tmp_path, capsys):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    result = force_settings_scale(str(settings_path), scale=13)

    assert result == {"foo": "bar"}
    assert "UYARI" in capsys.readouterr().out
    with open(settings_path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == {"foo": "bar"}


def test_build_multi_input_json_shapes_match_flattened_tensors():
    z = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    payload = build_multi_input_json(z, c)
    assert payload["input_data"][0] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert payload["input_data"][1] == [1.0, 0.0, 0.0, 1.0]


def test_write_json_file_then_load_bench_results_roundtrip(tmp_path):
    path = tmp_path / "sub" / "data.json"
    write_json_file({"a": 1}, str(path))
    assert load_bench_results(str(path)) == {"a": 1}


def test_load_bench_results_returns_empty_dict_when_missing(tmp_path):
    assert load_bench_results(str(tmp_path / "does_not_exist.json")) == {}


def test_save_bench_results_atomic_write_roundtrip(tmp_path):
    path = str(tmp_path / "bench_results.json")
    save_bench_results({"k1_matmul_scale8": {"status": "success"}}, path)
    assert load_bench_results(path) == {"k1_matmul_scale8": {"status": "success"}}
    # tmp dosyası kalıcı olarak kalmamalı (os.replace ile taşındı)
    assert not (tmp_path / "bench_results.json.tmp").exists()


def test_render_markdown_table_contains_header_and_rows():
    results = {
        "k1_matmul_scale8": {
            "status": "success",
            "timings": {"gen_settings": 1.0, "calibrate_settings": 2.0, "compile_circuit": 3.0, "setup": 4.0,
                        "gen_witness": 0.5, "prove": 5.0, "verify_offchain": 0.1},
            "proof_size_bytes": 1234,
            "pk_size_bytes": 111,
            "vk_size_bytes": 222,
            "peak_rss_kb": 500000,
            "decomposition_warning_count": 3,
            "max_abs_error": 0.0003,
            "deployed_bytecode_size": 15000,
            "exceeds_eip170": False,
            "verify_gas": 250000,
        },
        "k1_matmul_scale11": {"status": "failed", "error_summary": "RuntimeError: patladi", "timings": {}},
    }
    table = render_markdown_table(results)
    assert "kombinasyon" in table
    assert "k1_matmul_scale8" in table
    assert "k1_matmul_scale11" in table
    assert "RuntimeError: patladi" in table
    assert "success" in table and "failed" in table
