import torch

from circuits.toy_model import ToyMLP, make_example_input


def test_forward_output_shape():
    model = ToyMLP()
    x = make_example_input(batch_size=3)
    out = model(x)
    assert out.shape == (3, 8)


def test_param_count_matches_32_32_8():
    model = ToyMLP()
    # fc1: 32*32 + 32 = 1056, fc2: 32*8 + 8 = 264
    expected = (32 * 32 + 32) + (32 * 8 + 8)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected


def test_make_example_input_is_deterministic():
    a = make_example_input(2)
    b = make_example_input(2)
    assert torch.equal(a, b)
