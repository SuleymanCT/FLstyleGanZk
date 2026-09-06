"""Faz B: küçük MLP -> ONNX -> ezkl -> Solidity verifier -> anvil -> web3 doğrulama.

ezkl çağrı sırası (gen_settings -> calibrate_settings -> compile_circuit
-> get_srs -> setup -> gen_witness -> prove -> verify) aynı kullanıcının
`tez-projesi/src/zk.py` dosyasında `ezkl==23.0.5` ile Colab'da KANITLANMIŞ
sırayla birebir aynıdır. Faz B'nin ilk Colab koşumunda TÜM ZK adımları
(setup 9.1s, prove 13.5s, offchain verify 0.04s) geçti; tek hata EVM
adımında çıktı: `create_evm_verifier` de (get_srs gibi) coroutine
döndürüyor ama senkron çağrılmıştı ("RuntimeError: no running event
loop"). Bu yüzden HER ezkl çağrısı artık `circuits.ezkl_utils.run_async`
üzerinden geçiyor — hangi ezkl fonksiyonunun async olduğu sürümden
sürüme değişebildiği için (`get_srs` hep öyleydi, `create_evm_verifier`
23.0.5'te de öyle çıktı), bu sarmalayıcı sync/async ayrımını ÖNCEDEN
varsaymıyor, çalışma zamanında `inspect.isawaitable` ile kontrol ediyor.

BİLİNÇLİ TEST SINIRI: `ezkl`, `solc`, `anvil` gerektirir. Yerelde hiçbiri
yok — bu modül yerelde import bile edilemez (`import ezkl` başarısız
olur). `tests/test_toy_pipeline.py` bu yüzden modülü import etmeden önce
`pytest.importorskip("ezkl")` ile korur.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import ezkl
import torch

from chain.anvil import AnvilProcess
from chain.client import Web3Client
from chain.solc import compile_with_fallback_strategies
from circuits.ezkl_utils import run_async, run_get_srs

INPUT_VISIBILITY = "private"
PARAM_VISIBILITY = "fixed"  # ezkl'de "public" artık desteklenmiyor (deprecated)
OUTPUT_VISIBILITY = "public"


def export_onnx(model: torch.nn.Module, example_input: torch.Tensor, onnx_path: Path) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        example_input,
        str(onnx_path),
        dynamo=False,
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    if not onnx_path.exists():
        raise RuntimeError(f"torch.onnx.export sonrası dosya yok: {onnx_path}")


def write_input_json(example_input: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = example_input[:1].detach().cpu().numpy().reshape(-1).tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"input_data": [flat]}, f)


def run_ezkl_setup(onnx_path: Path, input_json_path: Path, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "settings": work_dir / "settings.json",
        "compiled": work_dir / "network.compiled",
        "pk": work_dir / "pk.key",
        "vk": work_dir / "vk.key",
    }

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = INPUT_VISIBILITY
    run_args.param_visibility = PARAM_VISIBILITY
    run_args.output_visibility = OUTPUT_VISIBILITY

    print("[toy_pipeline]  gen_settings...")
    ok = run_async(ezkl.gen_settings, str(onnx_path), str(paths["settings"]), py_run_args=run_args)
    if ok is not True:
        raise RuntimeError(f"gen_settings True döndürmedi: {ok!r}")

    print("[toy_pipeline]  calibrate_settings (target=resources)...")
    run_async(ezkl.calibrate_settings, str(input_json_path), str(onnx_path), str(paths["settings"]), "resources")

    print("[toy_pipeline]  compile_circuit...")
    ok = run_async(ezkl.compile_circuit, str(onnx_path), str(paths["compiled"]), str(paths["settings"]))
    if ok is not True:
        raise RuntimeError(f"compile_circuit True döndürmedi: {ok!r}")

    print("[toy_pipeline]  get_srs...")
    run_get_srs(str(paths["settings"]))

    print("[toy_pipeline]  setup (pk/vk üretiliyor)...")
    ok = run_async(ezkl.setup, str(paths["compiled"]), str(paths["vk"]), str(paths["pk"]))
    if ok is not True:
        raise RuntimeError(f"setup True döndürmedi: {ok!r}")

    for key, p in paths.items():
        if not p.exists():
            raise RuntimeError(f"setup sonrası beklenen '{key}' dosyası yok: {p}")

    return paths


def run_prove(input_json_path: Path, paths: dict, work_dir: Path) -> Path:
    witness_path = work_dir / "witness.json"
    proof_path = work_dir / "proof.json"

    print("[toy_pipeline]  gen_witness...")
    run_async(ezkl.gen_witness, str(input_json_path), str(paths["compiled"]), str(witness_path))
    if not witness_path.exists():
        raise RuntimeError(f"gen_witness sonrası witness dosyası yok: {witness_path}")

    print("[toy_pipeline]  prove...")
    run_async(ezkl.prove, str(witness_path), str(paths["compiled"]), str(paths["pk"]), str(proof_path))
    if not proof_path.exists():
        raise RuntimeError(f"prove sonrası proof dosyası yok: {proof_path}")

    return proof_path


def run_verify(proof_path: Path, paths: dict) -> bool:
    return bool(run_async(ezkl.verify, str(proof_path), str(paths["settings"]), str(paths["vk"])))


def generate_solidity_verifier(paths: dict, work_dir: Path) -> tuple[Path, Path]:
    sol_path = work_dir / "Verifier.sol"
    abi_path = work_dir / "Verifier.abi"
    run_async(ezkl.create_evm_verifier, str(paths["vk"]), str(paths["settings"]), str(sol_path), str(abi_path))
    if not sol_path.exists():
        raise RuntimeError(f"create_evm_verifier sonrası .sol dosyası yok: {sol_path}")
    return sol_path, abi_path


def compile_verifier_solidity(sol_path: Path) -> dict:
    """ezkl'nin ürettiği Verifier.sol yoğun assembly içerir; ne varsayılan
    solc ayarları ("Stack too deep") ne de tek başına `viaIR=True` +
    `optimizer_runs=200` (Faz B'nin 2. ve 3. Colab koşumlarında görüldü —
    3.'de "Cannot swap Variable ... too deep in the stack by 1 slots")
    her zaman yetiyor. `chain.solc.compile_with_fallback_strategies`
    birden fazla (solc sürümü × viaIR × optimizer_runs) kombinasyonunu
    sırayla dener, ilk çalışanı kullanır.
    """
    return compile_with_fallback_strategies(sol_path)


def deploy_and_verify_onchain(bytecode: bytes, proof_path: Path, work_dir: Path, anvil: AnvilProcess) -> dict:
    if not anvil.accounts:
        raise RuntimeError("anvil hesap listesi boş — deploy için hesap yok.")
    _, private_key = anvil.accounts[0]

    client = Web3Client(anvil.rpc_url)

    print("[toy_pipeline]  Verifier kontratı deploy ediliyor...")
    address, deploy_gas = client.deploy_bytecode(bytecode, private_key)
    print(f"[toy_pipeline]  Deploy edildi: {address} (gas={deploy_gas})")

    calldata_path = work_dir / "calldata.bin"
    print("[toy_pipeline]  encode_evm_calldata...")
    run_async(ezkl.encode_evm_calldata, str(proof_path), str(calldata_path))
    if not calldata_path.exists():
        raise RuntimeError(f"encode_evm_calldata sonrası calldata dosyası yok: {calldata_path}")
    calldata = calldata_path.read_bytes()

    print("[toy_pipeline]  Kontrata ispat gönderiliyor (verifyProof)...")
    verified, verify_gas = client.send_raw_call(address, calldata, private_key)

    return {
        "contract_address": address,
        "deploy_gas": deploy_gas,
        "verify_gas": verify_gas,
        "verified": verified,
    }


def deploy_and_verify_via_ezkl_native(sol_path: Path, proof_path: Path, work_dir: Path, anvil: AnvilProcess) -> dict:
    """SON ÇARE: `chain.solc.compile_with_fallback_strategies`'in TÜM
    kombinasyonları başarısız olursa, manuel solc derleme + web3 deploy
    yerine ezkl'nin KENDİ `deploy_evm`/`verify_evm` fonksiyonlarını
    kullanır — bunlar `sol_code_path`'i doğrudan alıp kendi iç solc
    çağrısıyla derleyip deploy/doğrulama yapıyor (resmi ezkl.pyi'de
    doğrulandı: `deploy_evm(addr_path, sol_code_path, rpc_url,
    contract_type, optimizer_runs, private_key)`,
    `verify_evm(addr_verifier, proof_path, rpc_url, vka_path)`).

    BİLİNÇLİ BELİRSİZLİK: `contract_type` parametresinin beklenen tam
    değeri (ör. "verifier") resmi dokümantasyonda açıklanmıyor — burada
    "verifier" deneniyor, yanlışsa ezkl'nin kendi hata mesajı burada
    (adım adı ile sarılı) görünecek. Bu yol kendi transaction'ını kendi
    gönderdiğinden `deploy_gas`/`verify_gas` receipt'ten doğrudan
    alınamıyor — `Web3Client.find_transaction_gas` ile geriye dönük
    zincir taramasıyla kurtarılmaya çalışılıyor, bulunamazsa `None`
    kalır (uydurulmaz).
    """
    if not anvil.accounts:
        raise RuntimeError("anvil hesap listesi boş — deploy için hesap yok.")
    _, private_key = anvil.accounts[0]
    client = Web3Client(anvil.rpc_url)

    addr_path = work_dir / "verifier_address.txt"
    print("[toy_pipeline]  ezkl.deploy_evm (native) ile deploy ediliyor...")
    run_async(ezkl.deploy_evm, str(addr_path), str(sol_path), anvil.rpc_url, "verifier", 200, private_key)
    if not addr_path.exists():
        raise RuntimeError(f"deploy_evm sonrası adres dosyası yok: {addr_path}")
    address = addr_path.read_text(encoding="utf-8").strip()
    print(f"[toy_pipeline]  Deploy edildi (native): {address}")

    deploy_gas = client.find_transaction_gas(contract_address=address)
    print(f"[toy_pipeline]  deploy_gas (geriye dönük bulundu): {deploy_gas}")

    print("[toy_pipeline]  ezkl.verify_evm (native) ile doğrulanıyor...")
    verified = bool(run_async(ezkl.verify_evm, address, str(proof_path), anvil.rpc_url, None))

    verify_gas = client.find_transaction_gas(to_address=address)
    print(f"[toy_pipeline]  verify_gas (geriye dönük bulundu): {verify_gas}")

    return {
        "contract_address": address,
        "deploy_gas": deploy_gas,
        "verify_gas": verify_gas,
        "verified": verified,
        "used_native_ezkl_deploy": True,
    }


def _run_step(report: dict, name: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - adım adını ekleyip yeniden fırlatıyoruz, yutmuyoruz
        raise RuntimeError(f"[adım: {name}] başarısız: {type(e).__name__}: {e}") from e
    elapsed = time.perf_counter() - t0
    report["steps"][name] = elapsed
    print(f"[toy_pipeline] '{name}' tamamlandı ({elapsed:.3f}s)")
    return result


def run_full_pipeline(work_dir: Path, anvil: AnvilProcess) -> dict:
    """Faz B'nin uçtan uca zincirini çalıştırır, adım süreleri + gas'ı
    içeren bir rapor döner. `anvil` çağıran tarafça (test) başlatılmış/
    kapatılacak bir `AnvilProcess` olmalı."""
    from circuits.toy_model import ToyMLP, make_example_input

    work_dir = Path(work_dir)
    report: dict = {"steps": {}}

    model = ToyMLP()
    example_input = make_example_input(batch_size=1)

    onnx_path = work_dir / "toy.onnx"
    input_json_path = work_dir / "input.json"

    _run_step(report, "export_onnx", export_onnx, model, example_input, onnx_path)
    _run_step(report, "write_input_json", write_input_json, example_input, input_json_path)
    ezkl_paths = _run_step(report, "ezkl_setup", run_ezkl_setup, onnx_path, input_json_path, work_dir)
    proof_path = _run_step(report, "ezkl_prove", run_prove, input_json_path, ezkl_paths, work_dir)
    off_chain_verified = _run_step(report, "ezkl_verify_offchain", run_verify, proof_path, ezkl_paths)
    if not off_chain_verified:
        raise RuntimeError("[adım: ezkl_verify_offchain] ezkl.verify False döndürdü — zincir dışı doğrulama başarısız.")

    sol_path, _abi_path = _run_step(report, "generate_solidity_verifier", generate_solidity_verifier, ezkl_paths, work_dir)

    try:
        compiled = _run_step(report, "compile_verifier_solidity", compile_verifier_solidity, sol_path)
    except RuntimeError as e:
        print(
            f"[toy_pipeline] TÜM solc stratejileri başarısız oldu, ezkl'nin kendi "
            f"deploy_evm/verify_evm yoluna (son çare) düşülüyor. Sebep:\n{e}"
        )
        report["solc_fallback_reason"] = str(e)
        onchain = _run_step(
            report, "deploy_and_verify_via_ezkl_native", deploy_and_verify_via_ezkl_native, sol_path, proof_path, work_dir, anvil
        )
        report.update(onchain)
        return report

    report["deployed_bytecode_size"] = compiled["deployed_bytecode_size"]
    report["exceeds_eip170"] = compiled["exceeds_eip170"]
    report["used_solc_strategy"] = compiled.get("used_strategy")
    if compiled["exceeds_eip170"]:
        print(
            "[toy_pipeline] UYARI: verifier'ın deployed bytecode'u EIP-170 sınırını aşıyor, "
            "deploy adımı başarısız olacaktır."
        )

    onchain = _run_step(
        report, "deploy_and_verify_onchain", deploy_and_verify_onchain, compiled["bytecode"], proof_path, work_dir, anvil
    )

    report.update(onchain)
    return report
