import pytest

from fl.shard_utils import build_shard_meta, extract_prefixed_state_dict
from storage.hashing import canonical_hash
from tests.fixtures import make_fake_state_dict


def test_extract_prefixed_state_dict_filters_expected_keys():
    full = make_fake_state_dict()
    shard = extract_prefixed_state_dict(full, ["mapping", "embed"])

    assert shard
    for key in shard:
        assert key.startswith("mapping") or key.startswith("embed")
    assert "synthesis_stub.weight" not in shard
    assert "synthesis_stub.bias" not in shard


def test_extract_prefixed_state_dict_raises_when_no_match():
    full = make_fake_state_dict()
    with pytest.raises(ValueError):
        extract_prefixed_state_dict(full, ["does_not_exist"])


def test_build_shard_meta_hash_matches_canonical_hash_directly():
    full = make_fake_state_dict()
    shard = extract_prefixed_state_dict(full, ["mapping", "embed"])
    meta = build_shard_meta(full, shard, extra={"round": 0, "site": 1})

    assert meta["full_generator_hash"] == canonical_hash(full)
    assert meta["round"] == 0
    assert meta["site"] == 1
    assert meta["shard_param_count"] < meta["full_param_count"]
    assert set(meta["shard_keys"]) == set(shard.keys())
