"""circuits/rebuild_mapping.py'nin doğruluğunu GERÇEK StyleGAN-XL
OLMADAN kanıtlar: sentetik rastgele ağırlıklarla bir sahte shard
kurulur, beklenen çıktı `rebuild_mapping`'in kendi sınıflarını
ÇAĞIRMADAN, doğrulanmış StyleGAN-XL formülüyle bağımsız olarak elle
hesaplanır, sonra `build_mapping_network_from_shard` + `forward`
çağrılıp ikisi karşılaştırılır.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from circuits.rebuild_mapping import (
    IGNORED_SHARD_KEYS,
    MinimalMappingNetwork,
    build_mapping_network_from_shard,
    build_pruned_mapping_network,
    compare_embed_modes,
    infer_dims_from_shard,
    verify_prunable_embed_rows,
)

Z_DIM = 4
EMBED_NUM = 10
EMBED_DIM = 6
W_DIM = 5
C_DIM = 3


def _make_fake_shard(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    shard = {
        "mapping.embed.weight": torch.randn(EMBED_NUM, EMBED_DIM, generator=g),
        "mapping.embed_proj.weight": torch.randn(Z_DIM, EMBED_DIM, generator=g),
        "mapping.embed_proj.bias": torch.randn(Z_DIM, generator=g),
        "mapping.fc0.weight": torch.randn(W_DIM, 2 * Z_DIM, generator=g),
        "mapping.fc0.bias": torch.randn(W_DIM, generator=g),
        "mapping.fc1.weight": torch.randn(W_DIM, W_DIM, generator=g),
        "mapping.fc1.bias": torch.randn(W_DIM, generator=g),
        "mapping.w_avg": torch.zeros(W_DIM),  # truncation buffer, kullanılmamalı
    }
    return shard


def _manual_eq_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, lr_multiplier: float) -> torch.Tensor:
    """rebuild_mapping'in kodunu ÇAĞIRMADAN, doğrulanmış formülün bağımsız
    bir elle-yazılmış kopyası (test bunun için var — iki bağımsız
    uygulamanın aynı sonucu verdiğini doğruluyor)."""
    weight_gain = lr_multiplier / math.sqrt(weight.shape[1])
    bias_gain = lr_multiplier
    w = weight * weight_gain
    b = bias * bias_gain
    x = x.matmul(w.t())
    x = F.leaky_relu(x + b, negative_slope=0.2) * math.sqrt(2.0)
    return x


def _manual_forward(shard: dict, z: torch.Tensor, c: torch.Tensor, embed_mode: str = "gather") -> torch.Tensor:
    x = z * (z.square().mean(dim=1, keepdim=True) + 1e-8).rsqrt()
    if embed_mode == "gather":
        indices = c.argmax(dim=1)
        embedded = shard["mapping.embed.weight"][indices]
    else:  # "matmul" — bagimsiz, ayri bir yol ile c @ embed.weight
        embedded = c.to(torch.float32).matmul(shard["mapping.embed.weight"])
    y = _manual_eq_linear(embedded, shard["mapping.embed_proj.weight"], shard["mapping.embed_proj.bias"], lr_multiplier=1.0)
    y = y * (y.square().mean(dim=1, keepdim=True) + 1e-8).rsqrt()
    x = torch.cat([x, y], dim=1)
    x = _manual_eq_linear(x, shard["mapping.fc0.weight"], shard["mapping.fc0.bias"], lr_multiplier=0.01)
    x = _manual_eq_linear(x, shard["mapping.fc1.weight"], shard["mapping.fc1.bias"], lr_multiplier=0.01)
    return x


def test_infer_dims_from_shard_correct():
    shard = _make_fake_shard()
    dims = infer_dims_from_shard(shard)
    assert dims == {"z_dim": Z_DIM, "embed_num": EMBED_NUM, "embed_dim": EMBED_DIM, "w_dim": W_DIM}


def test_infer_dims_from_shard_raises_on_missing_key():
    shard = _make_fake_shard()
    del shard["mapping.fc1.weight"]
    with pytest.raises(ValueError, match="eksik"):
        infer_dims_from_shard(shard)


def test_infer_dims_from_shard_raises_on_inconsistent_fc0_shape():
    shard = _make_fake_shard()
    shard["mapping.fc0.weight"] = torch.randn(W_DIM, 2 * Z_DIM + 1)  # tutarsız
    with pytest.raises(ValueError, match="fc0"):
        infer_dims_from_shard(shard)


def test_build_mapping_network_matches_independent_manual_computation():
    shard = _make_fake_shard(seed=42)
    model = build_mapping_network_from_shard(shard)
    model.eval()

    torch.manual_seed(7)
    k = 4
    z = torch.randn(k, Z_DIM)
    class_indices = torch.tensor([0, 1, 2, 0])
    c = F.one_hot(class_indices, num_classes=C_DIM).to(torch.float32)

    with torch.no_grad():
        rebuilt_output = model(z, c)
        expected_output = _manual_forward(shard, z, c)

    assert rebuilt_output.shape == (k, W_DIM)
    assert torch.allclose(rebuilt_output, expected_output, atol=1e-6)


def test_build_mapping_network_ignores_w_avg_but_logs_it(capsys):
    shard = _make_fake_shard()
    build_mapping_network_from_shard(shard)
    output = capsys.readouterr().out
    assert "mapping.w_avg" in output
    assert IGNORED_SHARD_KEYS == ("mapping.w_avg",)


def test_build_mapping_network_warns_on_unrecognized_key(capsys):
    shard = _make_fake_shard()
    shard["mapping.mystery_layer.weight"] = torch.randn(2, 2)
    build_mapping_network_from_shard(shard)
    output = capsys.readouterr().out
    assert "mapping.mystery_layer.weight" in output
    assert "UYARI" in output


def test_build_mapping_network_raises_on_shape_mismatch():
    shard = _make_fake_shard()
    shard["mapping.fc1.bias"] = torch.randn(W_DIM + 1)  # forward'a girmez ama sekil uyusmamali... use infer path
    with pytest.raises(ValueError):
        build_mapping_network_from_shard(shard)


def test_verify_prunable_embed_rows_accepts_in_range_indices():
    c = F.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=C_DIM).to(torch.float32)
    used = verify_prunable_embed_rows([c], num_classes=C_DIM)
    assert used == {0, 1, 2}


def test_verify_prunable_embed_rows_rejects_wrong_width():
    c_wrong_width = torch.zeros(4, C_DIM + 1)
    with pytest.raises(ValueError, match="şekli"):
        verify_prunable_embed_rows([c_wrong_width], num_classes=C_DIM)


def test_verify_prunable_embed_rows_multiple_tensors_union():
    c1 = F.one_hot(torch.tensor([0]), num_classes=C_DIM).to(torch.float32)
    c2 = F.one_hot(torch.tensor([2]), num_classes=C_DIM).to(torch.float32)
    used = verify_prunable_embed_rows([c1, c2], num_classes=C_DIM)
    assert used == {0, 2}


def test_build_pruned_mapping_network_matches_full_network_exactly():
    shard = _make_fake_shard(seed=99)
    full_model = build_mapping_network_from_shard(shard)
    full_model.eval()

    pruned_model = build_pruned_mapping_network(shard, num_classes=C_DIM)
    pruned_model.eval()

    assert pruned_model.embed.weight.shape == (C_DIM, EMBED_DIM)

    torch.manual_seed(3)
    k = 4
    z = torch.randn(k, Z_DIM)
    class_indices = torch.tensor([0, 1, 2, 0])
    c = F.one_hot(class_indices, num_classes=C_DIM).to(torch.float32)

    with torch.no_grad():
        full_output = full_model(z, c)
        pruned_output = pruned_model(z, c)

    assert torch.equal(full_output, pruned_output)


def test_build_pruned_mapping_network_rejects_num_classes_too_large():
    shard = _make_fake_shard()
    with pytest.raises(ValueError, match="büyük olamaz"):
        build_pruned_mapping_network(shard, num_classes=EMBED_NUM + 1)


def test_minimal_mapping_network_rejects_invalid_embed_mode():
    with pytest.raises(ValueError, match="embed_mode"):
        MinimalMappingNetwork(z_dim=Z_DIM, embed_num=EMBED_NUM, embed_dim=EMBED_DIM, w_dim=W_DIM, embed_mode="bogus")


def test_matmul_mode_matches_independent_manual_matmul_computation():
    shard = _make_fake_shard(seed=17)
    model = build_pruned_mapping_network(shard, num_classes=C_DIM, embed_mode="matmul")
    model.eval()

    pruned_shard = dict(shard)
    pruned_shard["mapping.embed.weight"] = shard["mapping.embed.weight"][:C_DIM].clone()

    torch.manual_seed(11)
    k = 4
    z = torch.randn(k, Z_DIM)
    class_indices = torch.tensor([0, 1, 2, 0])
    c = F.one_hot(class_indices, num_classes=C_DIM).to(torch.float32)

    with torch.no_grad():
        rebuilt_output = model(z, c)
        expected_output = _manual_forward(pruned_shard, z, c, embed_mode="matmul")

    assert torch.allclose(rebuilt_output, expected_output, atol=1e-6)


def test_gather_and_matmul_modes_match_within_tolerance():
    shard = _make_fake_shard(seed=5)
    gather_model = build_pruned_mapping_network(shard, num_classes=C_DIM, embed_mode="gather")
    matmul_model = build_pruned_mapping_network(shard, num_classes=C_DIM, embed_mode="matmul")
    gather_model.eval()
    matmul_model.eval()

    torch.manual_seed(21)
    k = 4
    z = torch.randn(k, Z_DIM)
    class_indices = torch.tensor([0, 1, 2, 0])
    c = F.one_hot(class_indices, num_classes=C_DIM).to(torch.float32)

    with torch.no_grad():
        gather_output = gather_model(z, c)
        matmul_output = matmul_model(z, c)

    assert torch.allclose(gather_output, matmul_output, atol=1e-6)


def test_compare_embed_modes_returns_small_diff_stats():
    shard = _make_fake_shard(seed=5)
    torch.manual_seed(21)
    k = 4
    z = torch.randn(k, Z_DIM)
    class_indices = torch.tensor([0, 1, 2, 0])
    c = F.one_hot(class_indices, num_classes=C_DIM).to(torch.float32)

    stats = compare_embed_modes(shard, num_classes=C_DIM, z=z, c=c, tolerance=1e-6)

    assert stats["max_abs_diff"] < 1e-6
    assert stats["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)


def test_compare_embed_modes_raises_when_tolerance_too_tight_for_noise():
    shard = _make_fake_shard(seed=5)
    torch.manual_seed(21)
    k = 4
    z = torch.randn(k, Z_DIM)
    class_indices = torch.tensor([0, 1, 2, 0])
    c = F.one_hot(class_indices, num_classes=C_DIM).to(torch.float32)

    # Gercek fark tipik olarak 0.0 (float32 kucuk boyutlarda tam esitlik
    # verebiliyor) - imkansiz derecede siki (negatif) bir tolerans ile
    # hatanin gercekten firladigini ve gercek degeri rapor ettigini dogrula.
    with pytest.raises(RuntimeError, match="max_abs_diff"):
        compare_embed_modes(shard, num_classes=C_DIM, z=z, c=c, tolerance=-1.0)
