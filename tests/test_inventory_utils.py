from fl.inventory_utils import (
    classify_pkl_paths,
    compare_training_options,
    parse_stats_jsonl,
)


def test_compare_training_options_finds_differing_key():
    options_by_path = {
        "round_0/site_0/training_options.json": {"lr": 0.001, "batch": 32},
        "round_0/site_1/training_options.json": {"lr": 0.002, "batch": 32},
    }
    result = compare_training_options(options_by_path)

    assert "lr" in result["differing_keys"]
    assert "batch" not in result["differing_keys"]
    assert result["num_files"] == 2


def test_compare_training_options_flags_missing_key_as_difference():
    options_by_path = {
        "a.json": {"lr": 0.001, "seed": 42},
        "b.json": {"lr": 0.001},
    }
    result = compare_training_options(options_by_path)

    assert "seed" in result["differing_keys"]
    assert result["differing_keys"]["seed"]["b.json"] == "<eksik>"


def test_parse_stats_jsonl_extracts_fid_series_and_skips_bad_lines():
    lines = [
        '{"step": 0, "fid50k_full": 120.5}',
        "not json at all",
        '{"step": 1, "fid50k_full": 90.1}',
        '{"step": 1, "unrelated_metric": 5}',
        "",
    ]
    result = parse_stats_jsonl(lines)

    assert result["num_lines"] == 5
    assert len(result["warnings"]) == 1
    assert result["warnings"][0]["satir_no"] == 2
    assert [p["value"] for p in result["fid_series"]["fid50k_full"]] == [120.5, 90.1]


def test_classify_pkl_paths_separates_expected_from_base_candidates():
    raw_root = "/content/drive/MyDrive/FL_Experiments/parallel_final"
    paths = [
        f"{raw_root}/round_0/site_0/00000-stylegan3-r-site0/network-snapshot.pkl",
        f"{raw_root}/round_3/site_2/00000-stylegan3-r-site2/network-snapshot.pkl",
        f"{raw_root}/base_model/network-snapshot.pkl",
        f"{raw_root}/network-snapshot-base.pkl",
    ]
    result = classify_pkl_paths(paths, raw_root)

    assert len(result["expected"]) == 2
    assert len(result["base_model_candidates"]) == 2
