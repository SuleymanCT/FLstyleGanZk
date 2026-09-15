"""chain/client.py testleri.

`chain.client` modülü kendi top-level'ında `web3`'ü import ettiğinden
(gerçek bir Web3Client her zaman canlı bir bağlantı ister), bu dosya
`web3` yerelde kurulu değilse SKIP olur (CLAUDE.md madde 7/8) — Colab'da
gerçekten koşar. `_build_tx` sentetik (sahte) bir `w3` nesnesiyle test
edilir; gerçek bir zincir bağlantısı gerekmez — bu, EIP-1559/legacy gas
alanlarının ASLA bir arada üretilmediğini kanıtlayan asıl test.
"""

import pytest

pytest.importorskip("web3", reason="web3 kurulu değil — chain/client.py testi sadece Colab'da (ya da web3 kurulu bir ortamda) koşar.")

from chain.client import _build_tx  # noqa: E402


class _FakeEth:
    def __init__(self, chain_id=31337, nonce=5, priority_fee=1_000_000_000, base_fee=875_000_000, gas_estimate=21000):
        self.chain_id = chain_id
        self.max_priority_fee = priority_fee
        self._nonce = nonce
        self._base_fee = base_fee
        self._gas_estimate = gas_estimate

    def get_transaction_count(self, address):
        return self._nonce

    def get_block(self, tag):
        assert tag == "latest"
        return {"baseFeePerGas": self._base_fee}

    def estimate_gas(self, tx):
        return self._gas_estimate


class _FakeW3:
    def __init__(self, **kwargs):
        self.eth = _FakeEth(**kwargs)


FROM_ADDRESS = "0x1111111111111111111111111111111111111111"


def _has_eip1559(tx: dict) -> bool:
    return "maxFeePerGas" in tx or "maxPriorityFeePerGas" in tx


def _has_legacy(tx: dict) -> bool:
    return "gasPrice" in tx


def test_build_tx_adds_eip1559_fields_when_none_present():
    tx = _build_tx(_FakeW3(), {}, FROM_ADDRESS)
    assert _has_eip1559(tx)
    assert not _has_legacy(tx)
    assert tx["maxPriorityFeePerGas"] == 1_000_000_000
    assert tx["maxFeePerGas"] == 875_000_000 * 2 + 1_000_000_000


def test_build_tx_respects_preexisting_eip1559_fields():
    # build_transaction()'ın zaten doldurmuş olabileceği durumu simüle eder.
    base_tx = {"maxFeePerGas": 999, "maxPriorityFeePerGas": 1}
    tx = _build_tx(_FakeW3(), base_tx, FROM_ADDRESS)
    assert tx["maxFeePerGas"] == 999
    assert tx["maxPriorityFeePerGas"] == 1
    assert not _has_legacy(tx)


def test_build_tx_respects_preexisting_legacy_field():
    tx = _build_tx(_FakeW3(), {"gasPrice": 123}, FROM_ADDRESS)
    assert tx["gasPrice"] == 123
    assert not _has_eip1559(tx)


def test_build_tx_raises_when_both_eip1559_and_legacy_present():
    with pytest.raises(ValueError, match="ASLA bir arada olmamalı"):
        _build_tx(_FakeW3(), {"gasPrice": 123, "maxFeePerGas": 999}, FROM_ADDRESS)


def test_build_tx_fills_from_nonce_chain_id_and_gas():
    tx = _build_tx(_FakeW3(), {}, FROM_ADDRESS)
    assert tx["from"] == FROM_ADDRESS
    assert tx["nonce"] == 5
    assert tx["chainId"] == 31337
    assert tx["gas"] == int(21000 * 1.2)


def test_build_tx_never_produces_both_gas_field_types_across_variants():
    variants = [
        {},
        {"maxFeePerGas": 500, "maxPriorityFeePerGas": 2},
        {"gasPrice": 500},
        {"data": "0xdeadbeef"},
    ]
    for base_tx in variants:
        tx = _build_tx(_FakeW3(), dict(base_tx), FROM_ADDRESS)
        assert not (_has_eip1559(tx) and _has_legacy(tx)), f"her ikisi de üretildi: {tx}"
        assert _has_eip1559(tx) or _has_legacy(tx), f"hiçbir gas alanı üretilmedi: {tx}"
