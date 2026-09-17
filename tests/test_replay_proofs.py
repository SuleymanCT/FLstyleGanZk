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
    check_rpc_alive,
    compute_mode_totals,
    get_replay_infra_config,
    parse_args,
    parse_int_range,
    render_failure_summary,
    render_infra_events,
    render_mode_comparison_table,
    render_staged_distribution,
    replay_combo_key,
    round_fully_done,
    should_restart_segment,
    validate_args,
)

_WORKER_ARGV = [
    "--worker",
    "--round-id", "0",
    "--site-index", "0",
    "--challenge-seed-hex", "ab",
    "--rpc-url", "http://127.0.0.1:8545",
    "--round-manager-address", "0xabc",
    "--round-manager-abi-file", "abi.json",
    "--private-key", "deadbeef",
    "--work-dir", "wd",
    "--result-out", "out.json",
]


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


def test_parse_args_worker_mode_does_not_require_mode():
    # Faz E'nin ilk Colab koşumunda tam BURADA patladı: --worker
    # çağrısı --mode göndermiyordu, argparse required=True yüzünden
    # "the following arguments are required: --mode" ile duruyordu.
    args = parse_args(_WORKER_ARGV)
    assert args.worker is True
    assert args.mode is None
    assert args.round_id == 0
    assert args.site_index == 0


def test_parse_args_parent_mode_mode_defaults_to_none_at_argparse_level():
    # required=True KALDIRILDI - argparse artık --mode eksikse HATA
    # VERMİYOR, zorunluluk validate_args'a taşındı (aşağıdaki testler).
    args = parse_args([])
    assert args.worker is False
    assert args.mode is None


def test_validate_args_passes_for_complete_worker_invocation():
    validate_args(parse_args(_WORKER_ARGV))  # raise etmemeli


def test_validate_args_raises_when_worker_missing_required_fields():
    with pytest.raises(ValueError, match="--worker ile şunlar da ZORUNLU"):
        validate_args(parse_args(["--worker"]))


def test_validate_args_lists_all_missing_worker_fields():
    with pytest.raises(ValueError) as exc_info:
        validate_args(parse_args(["--worker", "--round-id", "0"]))
    assert "--site-index" in str(exc_info.value)
    assert "--rpc-url" in str(exc_info.value)
    assert "--round-id" not in str(exc_info.value)  # bu VERİLDİ, eksik listesinde OLMAMALI


def test_validate_args_raises_when_mode_missing_in_parent_mode():
    with pytest.raises(ValueError, match="--mode ZORUNLU"):
        validate_args(parse_args([]))


def test_validate_args_passes_with_mode_in_parent_mode():
    validate_args(parse_args(["--mode", "full"]))  # raise etmemeli


def test_validate_args_logs_ignored_worker_args_in_parent_mode(capsys):
    validate_args(parse_args(["--mode", "full", "--round-id", "3"]))
    out = capsys.readouterr().out
    assert "--round-id" in out
    assert "YOKSAYILIYOR" in out


def test_get_replay_infra_config_reads_real_schedule_file():
    from orchestrator.schedule import load_schedule_config

    schedule_config = load_schedule_config("configs/schedule.yaml")
    infra = get_replay_infra_config(schedule_config)
    assert infra["anvil_restart_interval"] == 10
    assert infra["rpc_timeout_seconds"] == 120.0
    assert infra["health_check_timeout_seconds"] == 5.0


def test_get_replay_infra_config_falls_back_to_defaults_when_section_missing():
    infra = get_replay_infra_config({})
    assert infra["anvil_restart_interval"] == 10
    assert infra["rpc_timeout_seconds"] == 120.0
    assert infra["health_check_timeout_seconds"] == 5.0


def test_get_replay_infra_config_honors_explicit_overrides():
    infra = get_replay_infra_config({"replay_infra": {"anvil_restart_interval": 5}})
    assert infra["anvil_restart_interval"] == 5
    assert infra["rpc_timeout_seconds"] == 120.0  # override edilmeyen alan varsayılanda kalır


def test_should_restart_segment_true_when_threshold_reached():
    assert should_restart_segment(10, 10) is True
    assert should_restart_segment(11, 10) is True


def test_should_restart_segment_false_when_below_threshold():
    assert should_restart_segment(9, 10) is False


def test_should_restart_segment_disabled_when_interval_non_positive():
    assert should_restart_segment(1000, 0) is False
    assert should_restart_segment(1000, -1) is False


def test_round_fully_done_true_when_all_sites_present():
    all_results = {"round0_site0": {}, "round0_site1": {}, "round0_site2": {}, "round0_site3": {}}
    assert round_fully_done(0, [0, 1, 2, 3], all_results, force=False) is True


def test_round_fully_done_false_when_some_sites_missing():
    all_results = {"round0_site0": {}}
    assert round_fully_done(0, [0, 1, 2, 3], all_results, force=False) is False


def test_round_fully_done_false_when_force_true_even_if_all_present():
    all_results = {"round0_site0": {}, "round0_site1": {}, "round0_site2": {}, "round0_site3": {}}
    assert round_fully_done(0, [0, 1, 2, 3], all_results, force=True) is False


def test_check_rpc_alive_false_for_unreachable_port():
    # GERCEK bir ag cagrisi - hicbir sey dinlemeyen bir port'a - hizli
    # basarisiz olmali (baglanti reddi), False donmeli. Mock DEGIL.
    assert check_rpc_alive("http://127.0.0.1:1", timeout=2.0) is False


def test_check_rpc_alive_false_for_malformed_url():
    assert check_rpc_alive("not-a-url", timeout=2.0) is False


def test_render_infra_events_empty_reports_no_restarts():
    summary = render_infra_events([])
    assert "Hiç yeniden başlatma" in summary


def test_render_infra_events_lists_all_events_with_context():
    events = [
        {"type": "scheduled_restart", "segment_index": 0, "before_round": 3, "proofs_in_segment": 10},
        {"type": "unhealthy_before_proof", "segment_index": 1, "round_id": 9, "site_index": 2},
    ]
    summary = render_infra_events(events)
    assert "Toplam 2 altyapı olayı" in summary
    assert "scheduled_restart" in summary and "before_round=3" in summary
    assert "unhealthy_before_proof" in summary and "round_id=9" in summary and "site_index=2" in summary
