"""Minimal web3.py sarmalayıcısı: bytecode deploy + ham calldata gönderimi.

Faz B'nin oyuncak zincirinde (ezkl'nin ürettiği Solidity verifier'ı
deploy edip ispatı doğrulatmak için) ve Faz D'nin orkestratöründe
(RoundManager.sol ile etkileşim, gas ölçümü) ortak kullanılacak.

BİLİNÇLİ TEST SINIRI: gerçek bir EVM node'una (anvil) bağlantı gerektirir,
yerelde test edilemez — sadece Colab'da gerçek koşumda doğrulanabilir.
"""

from __future__ import annotations

from web3 import Web3


def _raw_transaction_bytes(signed_tx) -> bytes:
    # web3.py v6 -> v7 arasında SignedTransaction alan adı değişti
    # (rawTransaction -> raw_transaction). İkisini de dene.
    raw = getattr(signed_tx, "raw_transaction", None)
    if raw is None:
        raw = getattr(signed_tx, "rawTransaction", None)
    if raw is None:
        raise RuntimeError(
            "SignedTransaction nesnesinde ne 'raw_transaction' ne 'rawTransaction' "
            "bulundu — web3.py sürümü beklenenden farklı olabilir."
        )
    return raw


class Web3Client:
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"[Web3Client] '{rpc_url}'ye bağlanılamadı.")

    def _send_and_wait(self, tx: dict, private_key: str):
        account = self.w3.eth.account.from_key(private_key)
        tx = {**tx, "from": account.address}
        tx.setdefault("nonce", self.w3.eth.get_transaction_count(account.address))
        tx.setdefault("chainId", self.w3.eth.chain_id)
        tx.setdefault("gasPrice", self.w3.eth.gas_price)

        if "gas" not in tx:
            try:
                tx["gas"] = int(self.w3.eth.estimate_gas(tx) * 1.2)
            except Exception as e:  # noqa: BLE001 - tanılama amaçlı, net mesajla yeniden fırlatılıyor
                raise RuntimeError(f"[Web3Client] gas tahmini başarısız: {e}. tx={tx}") from e

        signed = account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(_raw_transaction_bytes(signed))
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        return receipt

    def deploy_bytecode(self, bytecode: bytes, private_key: str) -> tuple[str, int]:
        data = bytecode if isinstance(bytecode, str) else ("0x" + bytecode.hex())
        receipt = self._send_and_wait({"data": data}, private_key)
        if receipt["status"] != 1:
            raise RuntimeError(f"[Web3Client] Deploy başarısız (status=0). receipt={dict(receipt)}")
        address = receipt["contractAddress"]
        if not address:
            raise RuntimeError(f"[Web3Client] Deploy receipt'inde contractAddress yok. receipt={dict(receipt)}")
        return address, receipt["gasUsed"]

    def send_raw_call(self, to: str, data: bytes, private_key: str) -> tuple[bool, int]:
        """`data`'yı gerçek bir transaction olarak gönderir (eth_call DEĞİL —
        gerçek gasUsed ölçmek için). Bir transaction receipt'i, çağrının dönüş
        DEĞERİNİ içermez (sadece status/gasUsed/logs) — bu yüzden burada
        sadece (başarı, gasUsed) döner. Dönüş değerinin kendisi gerekiyorsa
        ayrıca `self.w3.eth.call(...)` ile simüle edilmeli."""
        hex_data = data if isinstance(data, str) else ("0x" + data.hex())
        receipt = self._send_and_wait({"to": to, "data": hex_data}, private_key)
        success = receipt["status"] == 1
        return success, receipt["gasUsed"]
