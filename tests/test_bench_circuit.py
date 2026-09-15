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
    classify_exception_status,
    cleanup_work_dir,
    combo_key,
    compute_dir_size_bytes,
    count_decomposition_warnings,
    default_work_root,
    extract_circuit_stats,
    extract_max_abs_error,
    load_bench_results,
    parse_combo_key,
    parse_markdown_table_column,
    parse_only_keys,
    read_realized_scale,
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


def test_parse_combo_key_is_inverse_of_combo_key():
    combo = {"k": 4, "embed_mode": "gather", "scale": 11}
    assert parse_combo_key(combo_key(combo)) == combo


def test_parse_combo_key_rejects_malformed_key():
    with pytest.raises(ValueError, match="Geçersiz kombinasyon anahtarı"):
        parse_combo_key("bogus_key")


def test_parse_combo_key_rejects_invalid_embed_mode():
    with pytest.raises(ValueError, match="embed_mode"):
        parse_combo_key("k4_bogus_scale11")


def test_parse_only_keys_parses_comma_separated_list():
    combos = parse_only_keys("k4_matmul_scale8, k1_matmul_scale8")
    assert combos == [
        {"k": 4, "embed_mode": "matmul", "scale": 8},
        {"k": 1, "embed_mode": "matmul", "scale": 8},
    ]


def test_parse_only_keys_rejects_empty_string():
    with pytest.raises(ValueError, match="--only"):
        parse_only_keys("")


def test_count_decomposition_warnings_counts_case_insensitive():
    log = "foo\nDecomposition error: integer 123 is too large\nbar\ndecomposition error: integer 999 is too large\n"
    assert count_decomposition_warnings(log) == 2


def test_count_decomposition_warnings_zero_when_absent():
    assert count_decomposition_warnings("her sey normal, uyari yok") == 0


def test_extract_max_abs_error_finds_value():
    log = "Numerical Fidelity Report: max_abs_error=0.00033\ndone"
    assert extract_max_abs_error(log) == pytest.approx(0.00033)


def test_parse_markdown_table_column_reads_value_below_header():
    log = (
        "Numerical Fidelity Report\n"
        "| mean_error | median_error | max_error | mean_abs_error | max_abs_error | mean_abs_percent_error |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| 0.001 | 0.0005 | 0.05 | 0.003 | 0.133 | 12.5 |\n"
    )
    assert parse_markdown_table_column(log, "max_abs_error") == pytest.approx(0.133)
    assert parse_markdown_table_column(log, "mean_abs_percent_error") == pytest.approx(12.5)


def test_parse_markdown_table_column_returns_none_when_column_absent():
    log = "| foo | bar |\n| --- | --- |\n| 1 | 2 |\n"
    assert parse_markdown_table_column(log, "max_abs_error") is None


def test_extract_max_abs_error_finds_value_in_real_pipe_table_format():
    log = (
        "[bench_circuit]  gen_witness...\n"
        "| max_abs_error | mean_abs_error |\n"
        "| --- | --- |\n"
        "| 0.133 | 0.02 |\n"
    )
    assert extract_max_abs_error(log) == pytest.approx(0.133)


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


def test_read_realized_scale_reads_run_args(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"run_args": {"input_scale": 8, "param_scale": 8, "logrows": 19, "other": "x"}}), encoding="utf-8")

    result = read_realized_scale(str(settings_path))

    assert result == {"input_scale": 8, "param_scale": 8, "logrows": 19}
    # dosyaya DOKUNULMAMALI (force_settings_scale'in aksine, artık sadece okuyoruz)
    with open(settings_path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["run_args"]["other"] == "x"


def test_read_realized_scale_warns_and_returns_empty_when_run_args_missing(tmp_path, capsys):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    result = read_realized_scale(str(settings_path))

    assert result == {}
    assert "UYARI" in capsys.readouterr().out


def test_classify_exception_status_detects_panic_by_type_name():
    class PanicException(BaseException):
        pass

    assert classify_exception_status(PanicException("left: 160, right: 144")) == "panic"


def test_classify_exception_status_returns_failed_for_normal_exceptions():
    assert classify_exception_status(RuntimeError("boom")) == "failed"


def test_default_work_root_returns_nonempty_local_path():
    root = default_work_root()
    assert isinstance(root, str) and root
    assert "zk_bench_work" in root


def test_compute_dir_size_bytes_sums_all_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 250)
    assert compute_dir_size_bytes(tmp_path) == 350


def test_cleanup_work_dir_deletes_only_configured_large_files(tmp_path):
    (tmp_path / "pk.key").write_bytes(b"p" * 1000)
    (tmp_path / "network.compiled").write_bytes(b"c" * 500)
    (tmp_path / "witness.json").write_bytes(b"w" * 200)
    (tmp_path / "settings.json").write_bytes(b"{}")
    (tmp_path / "proof.json").write_bytes(b"{}")

    result = cleanup_work_dir(tmp_path, keep_artifacts=False)

    assert not (tmp_path / "pk.key").exists()
    assert not (tmp_path / "network.compiled").exists()
    assert not (tmp_path / "witness.json").exists()
    assert (tmp_path / "settings.json").exists()
    assert (tmp_path / "proof.json").exists()
    assert set(result["cleaned_up_files"]) == {"pk.key", "network.compiled", "witness.json"}
    assert result["disk_usage_before_cleanup_bytes"] == 1704  # 1000+500+200+2+2
    assert result["disk_usage_after_cleanup_bytes"] == 4
    assert result["disk_usage_freed_bytes"] == 1700
    assert result["keep_artifacts"] is False


def test_cleanup_work_dir_keeps_everything_when_keep_artifacts_true(tmp_path):
    (tmp_path / "pk.key").write_bytes(b"p" * 1000)

    result = cleanup_work_dir(tmp_path, keep_artifacts=True)

    assert (tmp_path / "pk.key").exists()
    assert result["cleaned_up_files"] == []
    assert result["disk_usage_freed_bytes"] == 0
    assert result["keep_artifacts"] is True


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
            "realized_scale": {"input_scale": 8, "param_scale": 8, "logrows": 19},
            "timings": {"gen_settings": 1.0, "calibrate_settings": 2.0, "compile_circuit": 3.0, "setup": 4.0,
                        "gen_witness": 0.5, "prove": 5.0, "verify_offchain": 0.1},
            "proof_size_bytes": 1234,
            "pk_size_bytes": 111,
            "vk_size_bytes": 222,
            "peak_rss_kb": 500000,
            "decomposition_warning_count": 3,
            "max_abs_error": 0.133,
            "max_abs_error_relative_pct": 12.5,
            "disk_usage_before_cleanup_bytes": 2_600_000_000,
            "disk_usage_after_cleanup_bytes": 50_000,
            "deployed_bytecode_size": 15000,
            "exceeds_eip170": False,
            "verify_gas": 250000,
        },
        "k1_matmul_scale11": {"status": "failed", "error_summary": "RuntimeError: patladi", "timings": {}},
        "k1_matmul_scale16": {"status": "panic", "error_summary": "pyo3_runtime.PanicException: left: 160, right: 144", "timings": {}},
    }
    table = render_markdown_table(results)
    assert "kombinasyon" in table
    assert "gerceklesen_scale" in table
    assert "disk_oncesi(MB)" in table and "disk_sonrasi(MB)" in table
    assert "k1_matmul_scale8" in table
    assert "k1_matmul_scale11" in table
    assert "RuntimeError: patladi" in table
    assert "success" in table and "failed" in table and "panic" in table
    assert "12.5" in table
    assert "2600" in table  # disk_oncesi(MB), 2.6GB -> ~2600 MB
