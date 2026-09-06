import torch

import pytest

from scripts.audit_fedavg import load_fedavg_file


def test_load_fedavg_file_prefers_g_ema_over_g(tmp_path):
    path = tmp_path / "fedavg_0.pt"
    torch.save(
        {
            "G": {"mapping.fc0.weight": torch.tensor([100.0])},
            "G_ema": {"mapping.fc0.weight": torch.tensor([1.0]), "mapping.fc0.bias": torch.tensor([2.0])},
        },
        path,
    )

    result = load_fedavg_file(str(path))

    assert set(result.keys()) == {"mapping.fc0.weight", "mapping.fc0.bias"}
    assert torch.equal(result["mapping.fc0.weight"], torch.tensor([1.0]))  # G_ema'dan, G'den değil


def test_load_fedavg_file_falls_back_to_g_and_logs(tmp_path, capsys):
    path = tmp_path / "fedavg_0.pt"
    torch.save({"G": {"mapping.fc0.weight": torch.tensor([1.0])}}, path)

    result = load_fedavg_file(str(path))

    assert set(result.keys()) == {"mapping.fc0.weight"}
    assert torch.equal(result["mapping.fc0.weight"], torch.tensor([1.0]))
    assert "G_ema" in capsys.readouterr().out


def test_load_fedavg_file_raises_when_neither_g_nor_g_ema_present(tmp_path):
    path = tmp_path / "fedavg_0.pt"
    torch.save({"foo": {"a": torch.tensor([1.0])}}, path)

    with pytest.raises(KeyError):
        load_fedavg_file(str(path))


def test_load_fedavg_file_raises_for_non_tensor_value_inside_g_ema(tmp_path):
    path = tmp_path / "fedavg_0.pt"
    torch.save({"G_ema": {"a": torch.tensor([1.0]), "b": {"nested": "dict"}}}, path)

    with pytest.raises(TypeError, match=r"anahtar: 'b'"):
        load_fedavg_file(str(path))


def test_load_fedavg_file_raises_for_non_dict_top_level(tmp_path):
    path = tmp_path / "fedavg_0.pt"
    torch.save(torch.tensor([1.0, 2.0]), path)

    with pytest.raises(TypeError):
        load_fedavg_file(str(path))
