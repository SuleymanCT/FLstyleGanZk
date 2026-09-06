"""Faz B: küçük MLP -> ONNX -> ezkl -> Solidity verifier -> anvil -> web3 doğrulama.

ezkl çağrı sırası (gen_settings -> calibrate_settings -> compile_circuit
-> get_srs -> setup -> gen_witness -> prove -> verify) aynı kullanıcının
`tez-projesi/src/zk.py` dosyasında `ezkl==23.0.5` ile Colab'da KANITLANMIŞ
sırayla birebir aynıdır. EVM (create_evm_verifier/encode_evm_calldata)
kısmı resmi ezkl 23.0.5 Python binding dokümantasyonuna göre yazıldı ama
gerçek bir Colab koşumuyla henüz doğrulanmadı — bu yüzden her adım ayrı,
adı açık bir hataya sarılı (bir şey patlarsa TAM olarak hangi adımda
patladığı görülür).

BİLİNÇLİ TEST SINIRI: `ezkl`, `solc`, `anvil` gerektirir. Yerelde hiçbiri
yok — bu modül yerelde import bile edilemez (`import ezkl` başarısız
olur). `tests/test_toy_pipeline.py` bu yüzden modülü import etmeden önce
`pytest.importorskip("ezkl")` ile korur.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import ezkl
import torch

from chain.anvil import AnvilProcess
from chain.client import Web3Client
from circuits.ezkl_utils import run_get_srs

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
    ok = ezkl.gen_settings(str(onnx_path), str(paths["settings"]), py_run_args=run_args)
    if ok is not True:
        raise RuntimeError(f"gen_settings True döndürmedi: {ok!r}")

    print("[toy_pipeline]  calibrate_settings (target=resources)...")
    ezkl.calibrate_settings(str(input_json_path), str(onnx_path), str(paths["settings"]), "resources")

    print("[toy_pipeline]  compile_circuit...")
    ok = ezkl.compile_circuit(str(onnx_path), str(paths["compiled"]), str(paths["settings"]))
    if ok is not True:
        raise RuntimeError(f"compile_circuit True döndürmedi: {ok!r}")

    print("[toy_pipeline]  get_srs (async)...")
    run_get_srs(str(paths["settings"]))

    print("[toy_pipeline]  setup (pk/vk üretiliyor)...")
    ok = ezkl.setup(str(paths["compiled"]), str(paths["vk"]), str(paths["pk"]))
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
    ezkl.gen_witness(str(input_json_path), str(paths["compiled"]), str(witness_path))
    if not witness_path.exists():
        raise RuntimeError(f"gen_witness sonrası witness dosyası yok: {witness_path}")

    print("[toy_pipeline]  prove...")
    ezkl.prove(str(witness_path), str(paths["compiled"]), str(paths["pk"]), str(proof_path))
    if not proof_path.exists():
        raise RuntimeError(f"prove sonrası proof dosyası yok: {proof_path}")

    return proof_path


def run_verify(proof_path: Path, paths: dict) -> bool:
    return bool(ezkl.verify(str(proof_path), str(paths["settings"]), str(paths["vk"])))


def generate_solidity_verifier(paths: dict, work_dir: Path) -> tuple[Path, Path]:
    sol_path = work_dir / "Verifier.sol"
    abi_path = work_dir / "Verifier.abi"
    ezkl.create_evm_verifier(str(paths["vk"]), str(paths["settings"]), str(sol_path), str(abi_path))
    if not sol_path.exists():
        raise RuntimeError(f"create_evm_verifier sonrası .sol dosyası yok: {sol_path}")
    return sol_path, abi_path


def compile_verifier_solidity(sol_path: Path) -> bytes:
    result = subprocess.run(
        ["solc", "--combined-json", "bin", str(sol_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"solc başarısız (returncode={result.returncode}):\nstdout={result.stdout}\nstderr={result.stderr}"
        )

    data = json.loads(result.stdout)
    contracts = data.get("contracts", {})
    if not contracts:
        raise RuntimeError(f"solc çıktısında 'contracts' boş: {result.stdout[:2000]}")

    # Birden fazla kontrat olabilir (kütüphane/yardımcı kontratlar); en
    # büyük bytecode'a sahip olanı ana Verifier kontratı kabul ediyoruz.
    key, contract = max(contracts.items(), key=lambda kv: len(kv[1].get("bin", "")))
    bin_hex = contract.get("bin", "")
    if not bin_hex:
        raise RuntimeError(f"'{key}' için derlenmiş bytecode boş.")
    return bytes.fromhex(bin_hex)


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
    ezkl.encode_evm_calldata(str(proof_path), str(calldata_path))
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
    bytecode = _run_step(report, "compile_verifier_solidity", compile_verifier_solidity, sol_path)
    onchain = _run_step(report, "deploy_and_verify_onchain", deploy_and_verify_onchain, bytecode, proof_path, work_dir, anvil)

    report.update(onchain)
    return report
