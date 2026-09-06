import os

import pytest

from configs.loader import load_paths

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS_YAML = os.path.join(REPO_ROOT, "configs", "paths.yaml")


def test_local_block_has_expected_keys():
    paths = load_paths(env="local", config_path=PATHS_YAML)
    for key in ("raw_root", "zk_root", "shards_dir", "results_dir", "stylegan_xl_repo", "stylegan_train_root"):
        assert key in paths, f"local bloğunda '{key}' eksik"


def test_colab_block_has_expected_keys():
    paths = load_paths(env="colab", config_path=PATHS_YAML)
    for key in ("raw_root", "zk_root", "shards_dir", "results_dir", "stylegan_xl_repo", "stylegan_train_root"):
        assert key in paths, f"colab bloğunda '{key}' eksik"


def test_unknown_env_raises():
    with pytest.raises(ValueError):
        load_paths(env="staging", config_path=PATHS_YAML)


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_paths(env="local", config_path="does/not/exist.yaml")


def test_env_var_fallback(monkeypatch):
    monkeypatch.setenv("TEZ_ENV", "colab")
    paths = load_paths(config_path=PATHS_YAML)
    assert paths["raw_root"].startswith("/content/")
