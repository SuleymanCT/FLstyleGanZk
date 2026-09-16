"""scripts/replay_proofs.py testleri.

ezkl/anvil/solc/web3 GEREKTİRMEYEN saf yardımcı fonksiyonlar (aralık
ayrıştırma, kombinasyon anahtarlama, mod-toplamı hesaplama, markdown
tablo üretimi) burada GERÇEKTEN test edilir — modül kendi top-level'ında
`import ezkl` YAPMAZ (sadece `run_worker`/`main`'in ezkl/chain'e bağımlı
dallarında, çağrıldıklarında), bu yüzden bu dosya yerelde sorunsuz
çalışır.
"""

import pytest

from scripts.replay_proofs import (
    compute_mode_totals,
    parse_int_range,
    render_failure_summary,
    render_mode_comparison_table,
    render_staged_distribution,
    replay_combo_key,
)


def test_parse_int_range_simple_range():
    assert parse_int_range("0-14") == list(range(15))


def test_parse_int_range_comma_list():
    assert parse_int_range("1,3,5") == [1, 3, 5]


def test_parse_int_range_mixed_range_and_list():
    assert parse_int_range("0-2,5") == [0, 1, 2, 5]


def test_parse_int_range_single_value():
    assert parse_int_range("7") == [7]


def test_parse_int_range_rejects_empty():
    with pytest.raises(ValueError, match="Boş/geçersiz"):
        parse_int_range("")


def test_parse_int_range_rejects_backwards_range():
    with pytest.raises(ValueError, match="Geçersiz aralık"):
        parse_int_range("5-2")


def test_replay_combo_key_format():
    assert replay_combo_key(4, 2) == "round4_site2"


def test_compute_mode_totals_sums_across_proofs():
    results = {
        "round0_site0": {
            "status": "success",
            "timings": {"gen_settings": 1.0, "calibrate_settings": 2.0, "compile_circuit": 3.0, "setup": 4.0,
                        "gen_witness": 0.5, "prove": 5.0, "verify_offchain": 0.1, "solc_compile": 10.0},
            "deploy_gas": 2_900_000,
            "submit_gas": 1_100_000,
        },
        "round0_site1": {
            "status": "failed",
            "timings": {"gen_settings": 1.0},
            "deploy_gas": None,
            "submit_gas": None,
        },
    }
    totals = compute_mode_totals(results)
    assert totals["num_proofs"] == 2
    assert totals["num_success"] == 1
    assert totals["total_ezkl_seconds"] == pytest.approx(1.0 + 2.0 + 3.0 + 4.0 + 0.5 + 5.0 + 0.1 + 1.0)
    assert totals["total_solc_seconds"] == pytest.approx(10.0)
    assert totals["total_deploy_gas"] == 2_900_000
    assert totals["total_verify_gas"] == 1_100_000
    assert totals["total_chain_cost_gas"] == 2_900_000 + 1_100_000


def test_compute_mode_totals_handles_empty_results():
    totals = compute_mode_totals({})
    assert totals["num_proofs"] == 0
    assert totals["total_chain_cost_gas"] == 0


def test_render_mode_comparison_table_computes_savings_relative_to_full():
    totals_by_mode = {
        "full": {"num_proofs": 60, "num_success": 60, "total_ezkl_seconds": 4500.0, "total_solc_seconds": 1200.0,
                  "total_deploy_gas": 180_000_000, "total_verify_gas": 70_000_000, "total_chain_cost_gas": 250_000_000},
        "staged": {"num_proofs": 20, "num_success": 20, "total_ezkl_seconds": 1500.0, "total_solc_seconds": 400.0,
                    "total_deploy_gas": 60_000_000, "total_verify_gas": 23_000_000, "total_chain_cost_gas": 83_000_000},
    }
    table = render_mode_comparison_table(totals_by_mode)
    assert "full" in table and "staged" in table
    assert "60" in table  # ispat sayısı (full)
    # tasarruf: 1 - 83e6/250e6 = 0.668 -> %66.8
    assert "66.8" in table


def test_render_mode_comparison_table_savings_dash_when_full_missing():
    totals_by_mode = {"staged": {"num_proofs": 5, "num_success": 5, "total_ezkl_seconds": 1.0, "total_solc_seconds": 1.0,
                                   "total_deploy_gas": 1, "total_verify_gas": 1, "total_chain_cost_gas": 2}}
    table = render_mode_comparison_table(totals_by_mode)
    assert "| staged | 5 | 5 | 1.0 | 1.0 | 1 | 1 | 2 | - |" in table


def test_render_staged_distribution_groups_by_round():
    results = {
        "round0_site0": {"combo": {"round_id": 0, "site_index": 0}},
        "round0_site2": {"combo": {"round_id": 0, "site_index": 2}},
        "round7_site1": {"combo": {"round_id": 7, "site_index": 1}},
    }
    table = render_staged_distribution(results)
    lines = table.splitlines()
    assert any("| 0 | site_0, site_2 |" in line for line in lines)
    assert any("| 7 | site_1 |" in line for line in lines)


def test_render_staged_distribution_skips_entries_without_combo():
    assert render_staged_distribution({"bad": {}}) == "| round | ispat üreten site'lar |\n|---|---|\n"


def test_render_failure_summary_all_success_returns_clean_message():
    results = {"round0_site0": {"status": "success"}}
    assert render_failure_summary(results) == "Tüm kombinasyonlar başarılı — başarısızlık yok.\n"


def test_render_failure_summary_lists_failures_with_error_text():
    results = {
        "round0_site0": {"status": "success"},
        "round1_site2": {"status": "failed", "error_summary": "RuntimeError: kaboom"},
        "round2_site1": {"status": "crashed", "error": "returncode=-9"},
    }
    summary = render_failure_summary(results)
    assert "round0_site0" not in summary
    assert "round1_site2" in summary and "RuntimeError: kaboom" in summary
    assert "round2_site1" in summary and "returncode=-9" in summary


def test_render_failure_summary_escapes_pipes_and_newlines_in_error_text():
    results = {"round0_site0": {"status": "failed", "error_summary": "line1|with|pipes\nline2"}}
    summary = render_failure_summary(results)
    # tablo satırı bozulmamalı - '|' kaçırılmış, '\n' boşlukla değiştirilmiş olmalı
    assert summary.count("\n") == 3  # baslik + ayrac + tek veri satiri + son newline
    assert "line1\\|with\\|pipes line2" in summary
