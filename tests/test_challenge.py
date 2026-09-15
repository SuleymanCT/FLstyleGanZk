import pytest
import torch

from orchestrator.challenge import CHALLENGE_K, C_DIM, Z_DIM, build_challenge_z_c, challenge_seed_to_int


def _fake_seed(n: int) -> bytes:
    return n.to_bytes(32, byteorder="big")


def test_challenge_seed_to_int_is_deterministic():
    seed = _fake_seed(12345)
    assert challenge_seed_to_int(seed) == challenge_seed_to_int(seed)


def test_challenge_seed_to_int_differs_for_different_inputs():
    assert challenge_seed_to_int(_fake_seed(1)) != challenge_seed_to_int(_fake_seed(2))


def test_challenge_seed_to_int_rejects_non_bytes():
    with pytest.raises(TypeError):
        challenge_seed_to_int(12345)


def test_challenge_seed_to_int_rejects_empty_bytes():
    with pytest.raises(ValueError):
        challenge_seed_to_int(b"")


def test_build_challenge_z_c_same_seed_gives_same_z_c():
    seed = _fake_seed(999)
    z1, c1 = build_challenge_z_c(seed)
    z2, c2 = build_challenge_z_c(seed)
    assert torch.equal(z1, z2)
    assert torch.equal(c1, c2)


def test_build_challenge_z_c_different_seed_gives_different_z():
    z1, _c1 = build_challenge_z_c(_fake_seed(1))
    z2, _c2 = build_challenge_z_c(_fake_seed(2))
    assert not torch.equal(z1, z2)


def test_build_challenge_z_c_default_shapes_match_faz_c3_config():
    z, c = build_challenge_z_c(_fake_seed(7))
    assert z.shape == (CHALLENGE_K, Z_DIM)
    assert c.shape == (CHALLENGE_K, C_DIM)
    # c one-hot olmalı
    assert torch.equal(c.sum(dim=1), torch.ones(CHALLENGE_K))


def test_build_challenge_z_c_honors_custom_k():
    z, c = build_challenge_z_c(_fake_seed(7), k=4)
    assert z.shape == (4, Z_DIM)
    assert c.shape == (4, C_DIM)
