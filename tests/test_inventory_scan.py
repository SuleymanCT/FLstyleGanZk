import os

from scripts.inventory import (
    find_round_site_dirs,
    find_run_subdir,
    find_snapshot_pkl,
    scan_structure,
)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def _make_site(raw_root, round_idx, site_idx, run_name="00000-stylegan3-r-site", with_pkl=True):
    run_dir = os.path.join(raw_root, f"round_{round_idx}", f"site_{site_idx}", f"{run_name}{site_idx}")
    if with_pkl:
        _touch(os.path.join(run_dir, "network-snapshot.pkl"))
    _touch(os.path.join(run_dir, "log.txt"))
    _touch(os.path.join(run_dir, "stats.jsonl"))
    _touch(os.path.join(run_dir, "training_options.json"))
    return run_dir


def test_find_round_site_dirs_finds_all_pairs(tmp_path):
    raw_root = str(tmp_path)
    _make_site(raw_root, 0, 0)
    _make_site(raw_root, 0, 1)
    _make_site(raw_root, 1, 0)

    entries = find_round_site_dirs(raw_root)
    pairs = {(r, s) for r, s, _ in entries}
    assert pairs == {(0, 0), (0, 1), (1, 0)}


def test_find_run_subdir_reports_missing_and_multiple(tmp_path):
    raw_root = str(tmp_path)
    site_dir_empty = os.path.join(raw_root, "round_0", "site_0")
    os.makedirs(site_dir_empty)
    run_dir, warning = find_run_subdir(site_dir_empty)
    assert run_dir is None
    assert "bulunamadı" in warning

    site_dir_multi = os.path.join(raw_root, "round_0", "site_1")
    os.makedirs(os.path.join(site_dir_multi, "run_a"))
    os.makedirs(os.path.join(site_dir_multi, "run_b"))
    run_dir, warning = find_run_subdir(site_dir_multi)
    assert run_dir is None
    assert "birden fazla" in warning


def test_find_snapshot_pkl_prefers_exact_name(tmp_path):
    run_dir = str(tmp_path)
    _touch(os.path.join(run_dir, "network-snapshot.pkl"))
    path, warning = find_snapshot_pkl(run_dir)
    assert path == os.path.join(run_dir, "network-snapshot.pkl")
    assert warning is None


def test_find_snapshot_pkl_falls_back_and_warns(tmp_path):
    run_dir = str(tmp_path)
    _touch(os.path.join(run_dir, "network-snapshot-000010.pkl"))
    path, warning = find_snapshot_pkl(run_dir)
    assert path.endswith("network-snapshot-000010.pkl")
    assert warning is not None


def test_find_round_site_dirs_ignores_non_matching_site_suffix(tmp_path, capsys):
    raw_root = str(tmp_path)
    for round_idx in range(15):
        for site_idx in range(4):
            _make_site(raw_root, round_idx, site_idx)

    # site_0_test gibi ek, yanlış adlandırılmış bir "site" klasörü — sıkı
    # regex (^site_[0-3]$) buna uymaz, yoksayılmalı.
    bogus_dir = os.path.join(raw_root, "round_5", "site_0_test", "00000-stylegan3-r-site0")
    _touch(os.path.join(bogus_dir, "network-snapshot.pkl"))

    entries = find_round_site_dirs(raw_root)
    captured = capsys.readouterr()

    assert len(entries) == 60  # 61 değil
    assert "site_0_test" in captured.out
    assert "Yoksayıldı" in captured.out


def test_scan_structure_reports_missing_pairs_and_fedavg_files(tmp_path):
    raw_root = str(tmp_path)
    for round_idx in range(2):
        for site_idx in range(2):
            _make_site(raw_root, round_idx, site_idx)
    _touch(os.path.join(raw_root, "fedavg_0.pt"))
    _touch(os.path.join(raw_root, "fedavg_1.pt"))

    structure = scan_structure(raw_root)

    assert len(structure["sites"]) == 4
    assert (14, 3) in structure["missing_expected_pairs"]  # 15x4 grid beklenir, sadece 2x2 var
    assert len(structure["fedavg_files"]) == 2
    for site in structure["sites"]:
        assert site["files"]["network-snapshot.pkl"] is not None
