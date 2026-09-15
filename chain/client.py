"""Minimal web3.py sarmalayıcısı: bytecode deploy + ham calldata gönderimi
(`Web3Client`) ve `contracts/RoundManager.sol`'un her fonksiyonu için tip-
güvenli bir metod (`RoundManagerClient`, Faz D).

BİLİNÇLİ TEST SINIRI: gerçek bir EVM node'una (anvil) bağlantı gerektirir,
yerelde test edilemez — sadece Colab'da gerçek koşumda doğrulanabilir.
"""

from __future__ import annotations

from web3 import Web3

from chain.anvil import normalize_private_key_hex


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

    def deploy_contract(self, abi: list, bytecode: bytes, constructor_args: tuple, private_key: str) -> tuple[str, int]:
        """`Web3Client.deploy_bytecode`'un aksine constructor argümanlarını
        ABI'ye göre doğru şekilde kodlayıp bytecode'a ekler (web3.py'nin
        kendi `Contract.constructor(...)` mekanizmasıyla — elle ABI
        kodlaması YAPILMAZ, hataya açık olurdu). `contracts/RoundManager.sol`
        gibi constructor argümanlı kontratlar için kullanılır."""
        data = bytecode if isinstance(bytecode, str) else ("0x" + bytecode.hex())
        factory = self.w3.eth.contract(abi=abi, bytecode=data)
        # {} birak: build_transaction() "data"yı (bytecode + kodlanmis
        # constructor argumanlari) KENDISI hesaplar - elle vermek bunu EZER.
        tx = factory.constructor(*constructor_args).build_transaction({})
        receipt = self._send_and_wait(tx, private_key)
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

    def find_transaction_gas(self, *, contract_address: str | None = None, to_address: str | None = None) -> int | None:
        """Zinciri (geriye doğru) tarayıp verilen kritere uyan işlemin
        gasUsed'ini bulur. `ezkl.deploy_evm`/`verify_evm` gibi kendi
        transaction'ını kendi gönderen fonksiyonlardan SONRA gas rakamını
        geri kazanmak için kullanılır — yerel/izole bir anvil zincirinde
        tüm blokları taramak ucuzdur. Bulamazsa None döner (uydurmaz).
        """
        latest = self.w3.eth.block_number
        for block_num in range(latest, -1, -1):
            block = self.w3.eth.get_block(block_num, full_transactions=True)
            for tx in block["transactions"]:
                if contract_address is not None:
                    receipt = self.w3.eth.get_transaction_receipt(tx["hash"])
                    created = receipt.get("contractAddress")
                    if created and created.lower() == contract_address.lower():
                        return receipt["gasUsed"]
                elif to_address is not None and tx.get("to") and tx["to"].lower() == to_address.lower():
                    receipt = self.w3.eth.get_transaction_receipt(tx["hash"])
                    return receipt["gasUsed"]
        return None


class RoundManagerClient(Web3Client):
    """`contracts/RoundManager.sol`'un her fonksiyonu için bir metod. Her
    state-değiştiren (transaction gönderen) metod `(sonuç, gasUsed)` döner;
    görünüm (view) fonksiyonları gerçek bir transaction DEĞİL, `eth_call`
    olduğundan gas döndürmez.

    Private key'ler `normalize_private_key_hex` ile normalize edilip (0x
    öneksiz 64-hex doğrulaması) geri `0x` eklenerek web3.py'ye veriliyor —
    Faz C3'te `ezkl.deploy_evm`'in önek konusunda düştüğümüz tuzağa burada
    da düşmemek için (web3.py aslında ikisini de kabul eder, ama erken/net
    doğrulama için bilerek kullanılıyor)."""

    def __init__(self, rpc_url: str, address: str, abi: list):
        super().__init__(rpc_url)
        self.address = Web3.to_checksum_address(address)
        self.contract = self.w3.eth.contract(address=self.address, abi=abi)

    def _send(self, fn_call, private_key: str) -> tuple[dict, int]:
        private_key = "0x" + normalize_private_key_hex(private_key)
        account = self.w3.eth.account.from_key(private_key)
        tx = fn_call.build_transaction(
            {
                "from": account.address,
                "nonce": self.w3.eth.get_transaction_count(account.address),
                "chainId": self.w3.eth.chain_id,
                "gasPrice": self.w3.eth.gas_price,
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(_raw_transaction_bytes(signed))
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        if receipt["status"] != 1:
            raise RuntimeError(f"[RoundManagerClient] İşlem başarısız (status=0). receipt={dict(receipt)}")
        return receipt, receipt["gasUsed"]

    def register_site(self, site_address: str, private_key: str) -> int:
        _, gas = self._send(self.contract.functions.registerSite(Web3.to_checksum_address(site_address)), private_key)
        return gas

    def start_round(self, round_id: int, global_cid: str, global_hash: bytes, private_key: str) -> tuple[bytes, int]:
        _, gas = self._send(self.contract.functions.startRound(round_id, global_cid, global_hash), private_key)
        return self.get_challenge_seed(round_id), gas

    def submit_update(self, round_id: int, update_cid: str, weight_commitment: bytes, private_key: str) -> int:
        _, gas = self._send(self.contract.functions.submitUpdate(round_id, update_cid, weight_commitment), private_key)
        return gas

    def submit_proof(self, round_id: int, verifier_address: str, proof: bytes, public_inputs: list, private_key: str) -> tuple[bool, int]:
        """`verifier_address`: bu (round,site)'a özgü, `param_visibility="fixed"`
        yüzünden HER (round,site) için AYRI deploy edilmiş Verifier
        kontratının adresi (bkz. contracts/RoundManager.sol'un tepesindeki
        "ÖNEMLİ TASARIM NOTU")."""
        account = self.w3.eth.account.from_key("0x" + normalize_private_key_hex(private_key))
        _, gas = self._send(
            self.contract.functions.submitProof(round_id, Web3.to_checksum_address(verifier_address), proof, public_inputs),
            private_key,
        )
        verified = self.get_submission(round_id, account.address)["proof_verified"]
        return verified, gas

    def finalize_round(self, round_id: int, aggregate_cid: str, included_sites: list, private_key: str) -> int:
        checksummed = [Web3.to_checksum_address(s) for s in included_sites]
        _, gas = self._send(self.contract.functions.finalizeRound(round_id, aggregate_cid, checksummed), private_key)
        return gas

    def get_challenge_seed(self, round_id: int) -> bytes:
        return self.contract.functions.getChallengeSeed(round_id).call()

    def is_site_eligible(self, site_address: str) -> bool:
        return self.contract.functions.isSiteEligible(Web3.to_checksum_address(site_address)).call()

    def get_round_info(self, round_id: int) -> dict:
        global_cid, global_hash, challenge_seed, started_at, finalized, aggregate_cid = self.contract.functions.getRoundInfo(
            round_id
        ).call()
        return {
            "global_cid": global_cid,
            "global_hash": global_hash,
            "challenge_seed": challenge_seed,
            "started_at": started_at,
            "finalized": finalized,
            "aggregate_cid": aggregate_cid,
        }

    def get_submission(self, round_id: int, site_address: str) -> dict:
        update_cid, weight_commitment, submitted_at, proof_submitted, proof_verified, verifier_used = (
            self.contract.functions.getSubmission(round_id, Web3.to_checksum_address(site_address)).call()
        )
        return {
            "update_cid": update_cid,
            "weight_commitment": weight_commitment,
            "submitted_at": submitted_at,
            "proof_submitted": proof_submitted,
            "proof_verified": proof_verified,
            "verifier_used": verifier_used,
        }

    def get_reputation(self, site_address: str) -> int:
        return self.contract.functions.reputation(Web3.to_checksum_address(site_address)).call()

    def is_site_registered(self, site_address: str) -> bool:
        return self.contract.functions.registeredSites(Web3.to_checksum_address(site_address)).call()
