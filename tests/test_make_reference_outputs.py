import os

import pytest

from scripts.make_reference_outputs import parse_k_values, resolve_site_pkl_path


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def test_resolve_site_pkl_path_finds_pkl(tmp_path):
    raw_root = str(tmp_path)
    run_dir = os.path.join(raw_root, "round_14", "site_0", "00000-stylegan3-r-site0")
    _touch(os.path.join(run_dir, "network-snapshot.pkl"))

    pkl_path = resolve_site_pkl_path(raw_root, round_idx=14, site_idx=0)

    assert pkl_path == os.path.join(run_dir, "network-snapshot.pkl")


def test_resolve_site_pkl_path_raises_when_site_dir_missing(tmp_path):
    raw_root = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="Site dizini bulunamadı"):
        resolve_site_pkl_path(raw_root, round_idx=14, site_idx=0)


def test_resolve_site_pkl_path_raises_when_run_subdir_ambiguous(tmp_path):
    raw_root = str(tmp_path)
    site_dir = os.path.join(raw_root, "round_14", "site_0")
    os.makedirs(os.path.join(site_dir, "run_a"))
    os.makedirs(os.path.join(site_dir, "run_b"))

    with pytest.raises(RuntimeError, match="birden fazla"):
        resolve_site_pkl_path(raw_root, round_idx=14, site_idx=0)


def test_resolve_site_pkl_path_raises_when_pkl_missing(tmp_path):
    raw_root = str(tmp_path)
    run_dir = os.path.join(raw_root, "round_14", "site_0", "00000-stylegan3-r-site0")
    os.makedirs(run_dir)

    with pytest.raises(FileNotFoundError, match="pkl bulunamadı"):
        resolve_site_pkl_path(raw_root, round_idx=14, site_idx=0)


def test_parse_k_values_parses_comma_separated_ints():
    assert parse_k_values("1,4,8") == [1, 4, 8]


def test_parse_k_values_strips_whitespace():
    assert parse_k_values(" 1, 4 ,8 ") == [1, 4, 8]


def test_parse_k_values_raises_on_empty():
    with pytest.raises(ValueError):
        parse_k_values("")
