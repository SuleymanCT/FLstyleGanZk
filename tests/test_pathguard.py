import os

import pytest

from storage.pathguard import PROTECTED_MARKERS, assert_writable, open_readonly


@pytest.mark.parametrize("marker", PROTECTED_MARKERS)
def test_assert_writable_raises_for_each_protected_marker(tmp_path, marker):
    protected_path = os.path.join(str(tmp_path), marker, "sub", "out.json")
    with pytest.raises(RuntimeError):
        assert_writable(protected_path)


def test_assert_writable_allows_unrelated_path(tmp_path):
    allowed_path = os.path.join(str(tmp_path), "zk_artifacts_results", "inventory.json")
    assert_writable(allowed_path)  # raise etmemeli


def test_assert_writable_catches_marker_deep_in_relative_path():
    with pytest.raises(RuntimeError):
        assert_writable("some/relative/FL_Experiments/parallel_final/round_0/x.pt")


def test_open_readonly_reads_existing_file_in_binary_mode(tmp_path):
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"hello world")

    handle = open_readonly(str(file_path))
    try:
        assert "b" in handle.mode
        assert handle.read() == b"hello world"
    finally:
        handle.close()


def test_open_readonly_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        open_readonly("does/not/exist.bin")
