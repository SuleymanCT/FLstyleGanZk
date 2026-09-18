"""attacks/detection.py testleri — saf, StyleGAN-XL/GPU gerektirmez."""

import torch

from attacks.detection import attack_touches_zk_proven_scope, verify_commitment_consistency


def test_verify_commitment_consistency_true_for_identical_state():
    state = {"mapping.fc0.weight": torch.arange(6.0).reshape(2, 3)}
    assert verify_commitment_consistency(state, dict(state)) is True


def test_verify_commitment_consistency_false_for_different_state():
    a = {"mapping.fc0.weight": torch.zeros(2, 3)}
    b = {"mapping.fc0.weight": torch.ones(2, 3)}
    assert verify_commitment_consistency(a, b) is False


def test_attack_touches_zk_proven_scope_true_when_mapping_key_present():
    assert attack_touches_zk_proven_scope(["mapping.embed.weight", "synthesis.const"]) is True


def test_attack_touches_zk_proven_scope_false_when_only_synthesis_keys():
    assert attack_touches_zk_proven_scope(["synthesis.const", "synthesis.L0.weight"]) is False


def test_attack_touches_zk_proven_scope_false_for_empty_list():
    assert attack_touches_zk_proven_scope([]) is False
