import inspect

import torch

from fl.reference_utils import build_class_indices, build_z_c, resolve_mapping_kwargs


def test_build_class_indices_k_less_than_c_dim():
    assert build_class_indices(3, 5) == [0, 1, 2]


def test_build_class_indices_k_greater_than_c_dim_cycles():
    assert build_class_indices(8, 5) == [0, 1, 2, 3, 4, 0, 1, 2]


def test_build_class_indices_k_equal_c_dim():
    assert build_class_indices(5, 5) == [0, 1, 2, 3, 4]


def test_build_z_c_shapes():
    z, c = build_z_c(k=4, z_dim=64, c_dim=5, seed=0)
    assert z.shape == (4, 64)
    assert c.shape == (4, 5)


def test_build_z_c_c_is_one_hot():
    z, c = build_z_c(k=8, z_dim=64, c_dim=5, seed=0)
    assert torch.all(c.sum(dim=1) == 1.0)
    expected_indices = torch.tensor(build_class_indices(8, 5))
    assert torch.equal(c.argmax(dim=1), expected_indices)


def test_build_z_c_same_seed_is_deterministic():
    z1, c1 = build_z_c(4, 64, 5, seed=42)
    z2, c2 = build_z_c(4, 64, 5, seed=42)
    assert torch.equal(z1, z2)
    assert torch.equal(c1, c2)


def test_build_z_c_different_seed_differs():
    z1, _ = build_z_c(4, 64, 5, seed=0)
    z2, _ = build_z_c(4, 64, 5, seed=1)
    assert not torch.equal(z1, z2)


def test_build_z_c_does_not_pollute_global_rng_state():
    torch.manual_seed(123)
    before = torch.randn(3)

    torch.manual_seed(123)
    build_z_c(4, 64, 5, seed=999)
    after = torch.randn(3)

    assert torch.equal(before, after)


def test_resolve_mapping_kwargs_includes_when_present():
    def fake_forward(self, z, c, truncation_psi=1, truncation_cutoff=None):
        pass

    sig = inspect.signature(fake_forward)
    kwargs = resolve_mapping_kwargs(sig, truncation_psi=1.0, truncation_cutoff=None)
    assert kwargs == {"truncation_psi": 1.0, "truncation_cutoff": None}


def test_resolve_mapping_kwargs_omits_when_absent():
    def fake_forward(self, z, c):
        pass

    sig = inspect.signature(fake_forward)
    kwargs = resolve_mapping_kwargs(sig, truncation_psi=1.0, truncation_cutoff=None)
    assert kwargs == {}


def test_resolve_mapping_kwargs_partial_presence():
    def fake_forward(self, z, c, truncation_psi=1):
        pass

    sig = inspect.signature(fake_forward)
    kwargs = resolve_mapping_kwargs(sig, truncation_psi=0.7, truncation_cutoff=8)
    assert kwargs == {"truncation_psi": 0.7}
