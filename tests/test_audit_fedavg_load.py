import torch

import pytest

from scripts.audit_fedavg import load_fedavg_file


def test_load_fedavg_file_handles_flat_state_dict(tmp_path):
    path = tmp_path / "fedavg_flat.pt"
    torch.save({"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}, path)

    result = load_fedavg_file(str(path))

    assert set(result.keys()) == {"a", "b"}
    assert result["a"].dtype == torch.float32
    assert torch.equal(result["b"], torch.tensor([3.0]))


def test_load_fedavg_file_unwraps_single_key_wrapper(tmp_path, capsys):
    path = tmp_path / "fedavg_wrapped.pt"
    torch.save({"G_ema": {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}}, path)

    result = load_fedavg_file(str(path))

    assert set(result.keys()) == {"a", "b"}
    assert "sarmalayıcı" in capsys.readouterr().out


def test_load_fedavg_file_unwraps_two_nested_wrappers(tmp_path):
    path = tmp_path / "fedavg_double_wrapped.pt"
    torch.save({"outer": {"inner": {"a": torch.tensor([1.0])}}}, path)

    result = load_fedavg_file(str(path))

    assert set(result.keys()) == {"a"}


def test_load_fedavg_file_raises_clear_error_for_ambiguous_nested_dict(tmp_path):
    path = tmp_path / "fedavg_ambiguous.pt"
    torch.save({"G": {"a": torch.tensor([1.0])}, "D": {"b": torch.tensor([2.0])}}, path)

    with pytest.raises(TypeError, match=r"anahtar: 'G'"):
        load_fedavg_file(str(path))


def test_load_fedavg_file_raises_for_non_dict_top_level(tmp_path):
    path = tmp_path / "fedavg_not_dict.pt"
    torch.save(torch.tensor([1.0, 2.0]), path)

    with pytest.raises(TypeError):
        load_fedavg_file(str(path))
