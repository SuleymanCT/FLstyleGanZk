#!/usr/bin/env python
"""Faz D: `contracts/RoundManager.sol`'u deploy eder, sonucu bir JSON
dosyasına yazar (`orchestrator/round_runner.py` bunu okuyup kontratla
konuşur).

**Burada bir Verifier.sol DEPLOY EDİLMİYOR** — `RoundManager.sol`'un
tepesindeki "ÖNEMLİ TASARIM NOTU"na bakın: ezkl `param_visibility="fixed"`
kullandığından (Faz B/C boyunca), ağırlıklar devrenin sabit sütunlarına
gömülü — her (round,site)'ın ağırlıkları FARKLI olduğundan HER ispatın
KENDİ Verifier'ı olur. Bu yüzden Verifier deploy'u `round_runner.py`'ye
(her gerekli ispat için TAZE) bırakıldı, burada TEK SEFERLİK olan
`RoundManager` deploy ediliyor.

İtibar parametreleri `configs/schedule.yaml`'dan okunur — koda gömülü
değil. `RoundManager.sol` derlemesi `chain.solc.compile_with_fallback_strategies`
(Faz B'de doğrulanan viaIR=False/runs=200/solc=0.8.20 ilk sırada) DOĞRUDAN
yeniden kullanılıyor.

**anvil bu script tarafından BAŞLATILMIYOR/YÖNETİLMİYOR** — bilerek:
deploy edilen `RoundManager`, script bittikten SONRA da (round_runner.py'nin
sonraki çağrılarında) AYNI zincirde durmalı; script kendi kısa ömürlü
`AnvilProcess`'ini açıp kapatsaydı deploy ettiği şey onunla birlikte yok
olurdu. Bu yüzden `--rpc-url`/`--private-key` ZORUNLU — anvil (ya da
başka bir EVM node'u) ayrıca, bu script'ten BAĞIMSIZ başlatılmalı (ör.
Colab'da bir hücrede `anvil --port 8545 &`).

BİLİNÇLİ TEST SINIRI: solc+web3 gerektirir, sadece Colab'da çalışır.
Sadece `--help` ve saf `build_deployment_record` yerelde test edilir.

Kullanım (Colab'da, anvil ayrıca başlatılmış olmalı):
    anvil --port 8545 &
    python -m scripts.deploy_contracts --env colab \
        --rpc-url http://127.0.0.1:8545 --private-key <anvil_hesabi>
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from configs.loader import load_paths
from orchestrator.schedule import load_schedule_config
from scripts.bench_circuit import write_json_file


def build_deployment_record(round_manager_address: str, round_manager_abi: list, rpc_url: str) -> dict:
    return {
        "round_manager_address": round_manager_address,
        "round_manager_abi": round_manager_abi,
        "rpc_url": rpc_url,
        "deployed_at": time.time(),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz D: RoundManager.sol'u deploy et.")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--schedule-config", default="configs/schedule.yaml")
    parser.add_argument("--round-manager-sol", default="contracts/RoundManager.sol")
    parser.add_argument("--rpc-url", required=True, help="anvil (ya da başka bir EVM node'u) ZATEN çalışıyor olmalı")
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--output", default=None, help="varsayılan: {zk_root}/chain/deployment.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from chain.client import Web3Client
    from chain.solc import compile_with_fallback_strategies

    paths = load_paths(env=args.env, config_path=args.paths_config)
    schedule_config = load_schedule_config(args.schedule_config)

    print(f"[deploy_contracts] '{args.round_manager_sol}' derleniyor...")
    compiled = compile_with_fallback_strategies(Path(args.round_manager_sol))
    if compiled["exceeds_eip170"]:
        raise RuntimeError(
            f"RoundManager deployed bytecode EIP-170'i AŞIYOR ({compiled['deployed_bytecode_size']} byte) — deploy edilemez."
        )

    proof_schedule = schedule_config["proof_schedule"]
    constructor_args = (
        schedule_config["reputation_initial"],
        schedule_config["reputation_penalty"],
        schedule_config["reputation_bonus"],
        proof_schedule["reputation_threshold"],
    )

    client = Web3Client(args.rpc_url)
    print(f"[deploy_contracts] RoundManager deploy ediliyor (constructor_args={constructor_args})...")
    round_manager_address, gas = client.deploy_contract(compiled["abi"], compiled["bytecode"], constructor_args, args.private_key)
    print(f"[deploy_contracts] RoundManager deploy edildi: {round_manager_address} (gas={gas})")

    record = build_deployment_record(round_manager_address, compiled["abi"], args.rpc_url)
    output_path = args.output or os.path.join(paths["zk_root"], "chain", "deployment.json")
    write_json_file(record, output_path)
    print(f"[deploy_contracts] Yazıldı: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
