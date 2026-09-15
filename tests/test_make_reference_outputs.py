import os

import pytest
import torch
import torch.nn as nn

from scripts.make_reference_outputs import parse_k_values, register_intermediate_hooks, resolve_site_pkl_path


class _FakeMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_proj = nn.Linear(3, 4)
        self.fc0 = nn.Linear(4, 5)
        self.fc1 = nn.Linear(5, 5)

    def forward(self, x):
        return self.fc1(self.fc0(self.embed_proj(x)))


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


def test_register_intermediate_hooks_captures_submodule_outputs():
    mapping = _FakeMapping()
    captured, handles = register_intermediate_hooks(mapping)

    x = torch.randn(2, 3)
    with torch.no_grad():
        mapping(x)
    for h in handles:
        h.remove()

    assert set(captured.keys()) == {"embed_proj_out", "fc0_out", "fc1_out"}
    assert captured["embed_proj_out"].shape == (2, 4)
    assert captured["fc0_out"].shape == (2, 5)
    assert captured["fc1_out"].shape == (2, 5)


def test_register_intermediate_hooks_warns_on_missing_submodule(capsys):
    class _PartialMapping(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc0 = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.fc0(x)

    mapping = _PartialMapping()
    captured, handles = register_intermediate_hooks(mapping)
    for h in handles:
        h.remove()

    output = capsys.readouterr().out
    assert "embed_proj" in output
    assert "fc1" in output
    assert "fc0_out" not in captured  # hook hiç calismadi, cunku forward cagrilmadi
