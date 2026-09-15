import pytest

from storage.ipfs import get, is_valid_cid


def test_is_valid_cid_accepts_real_cidv0_example():
    # kubo dokumantasyonundaki bilinen ornek CIDv0
    assert is_valid_cid("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG") is True


def test_is_valid_cid_accepts_cidv1_example():
    assert is_valid_cid("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi") is True


def test_is_valid_cid_rejects_garbage():
    assert is_valid_cid("not a cid") is False
    assert is_valid_cid("") is False
    assert is_valid_cid(None) is False


def test_get_rejects_invalid_cid_before_touching_subprocess(tmp_path):
    with pytest.raises(ValueError, match="Geçersiz CID"):
        get("not-a-cid", str(tmp_path / "out"))
