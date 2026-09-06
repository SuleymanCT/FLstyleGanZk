"""Standard-JSON ile `solc` derleme.

ezkl'nin ürettiği Verifier.sol gibi assembly-ağır kontratlar varsayılan
solc ayarlarıyla "Stack too deep" hatası verebilir — `viaIR` +
optimizer gerekir. Bu modül Faz B'nin oyuncak verifier'ı VE Faz D'nin
`RoundManager.sol`'u tarafından ortak kullanılacak.

`_parse_standard_json_output` SAF bir fonksiyondur (solc'un ürettiği
JSON'u alır, hata/kontrat seçimi/EIP-170 kontrolü yapar) ve sentetik
JSON'larla test edilir. `compile_solidity` (gerçek `solc` subprocess'ini
çalıştıran kısım) BİLİNÇLİ TEST SINIRI içindedir — solc yerelde yok.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

EIP170_MAX_DEPLOYED_BYTECODE_SIZE = 24576
DEFAULT_OPTIMIZER_RUNS = 200
DEFAULT_EVM_VERSION = "shanghai"
DEFAULT_TIMEOUT_SECONDS = 300.0

# Denenecek (via_ir, optimizer_runs) kombinasyonları. Faz B'nin gerçek
# Colab koşumunda ezkl==23.0.5'in ürettiği Halo2Verifier.sol için
# `viaIR=False, optimizer_runs=200` çalıştı (solc=0.8.20) — hata
# mesajının önerdiğinin (--via-ir) TERSİ: via-IR'in kendi Yul
# optimizer'ı bu yoğun elle yazılmış assembly'de AYRI bir "stack too
# deep" (YulException) çıkardı, klasik/legacy codegen + optimizer
# sorunsuz derledi (bkz. docs/phase_b_report.md). Bu yüzden en olası
# çalışan kombinasyon ilk sırada — sonraki koşularda 3 gereksiz
# başarısız denemeyle zaman kaybetmemek için.
DEFAULT_STRATEGIES = [
    {"via_ir": False, "optimizer_runs": 200},
    {"via_ir": True, "optimizer_runs": 1},
    {"via_ir": True, "optimizer_runs": 50},
    {"via_ir": True, "optimizer_runs": 200},
]

# Her strateji bu solc sürümlerinin her biriyle denenir (sürüm başına
# solc-select ile geçiş yapılır). ezkl==23.0.5'in ürettiği pragma'ya
# göre hepsi uygun olmayabilir — o yüzden pragma/ilk satırlar loglanır.
DEFAULT_SOLC_VERSIONS = ["0.8.20", "0.8.24", "0.8.26"]


def build_standard_json_input(
    source_name: str,
    source_code: str,
    optimizer_runs: int = DEFAULT_OPTIMIZER_RUNS,
    via_ir: bool = True,
    evm_version: str = DEFAULT_EVM_VERSION,
) -> dict:
    return {
        "language": "Solidity",
        "sources": {source_name: {"content": source_code}},
        "settings": {
            "optimizer": {"enabled": True, "runs": optimizer_runs},
            "viaIR": via_ir,
            "evmVersion": evm_version,
            "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"]}},
        },
    }


def format_solc_messages(entries: list[dict]) -> str:
    lines = []
    for entry in entries:
        lines.append(entry.get("formattedMessage") or entry.get("message") or json.dumps(entry))
    return "\n".join(lines)


def read_source_header(sol_path: Path, num_lines: int = 20) -> str:
    """Bir .sol dosyasının ilk `num_lines` satırını döner (pragma/sürüm
    bilgisini görmek için — hangi solc sürümünün beklendiğini anlamaya
    yarar)."""
    sol_path = Path(sol_path)
    lines: list[str] = []
    with open(sol_path, "r", encoding="utf-8") as f:
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line)
    return "".join(lines)


def _parse_standard_json_output(output: dict) -> dict:
    all_entries = output.get("errors", [])
    fatal_errors = [e for e in all_entries if e.get("severity") == "error"]
    warnings = [e for e in all_entries if e.get("severity") != "error"]

    if fatal_errors or "contracts" not in output:
        raise RuntimeError(
            f"[solc] Derleme başarısız ({len(fatal_errors)} hata):\n{format_solc_messages(fatal_errors)}"
        )

    contracts = output["contracts"]
    flat = [
        (f"{file_name}:{contract_name}", contract_data)
        for file_name, file_contracts in contracts.items()
        for contract_name, contract_data in file_contracts.items()
    ]
    if not flat:
        raise RuntimeError(f"[solc] Derlenmiş kontrat bulunamadı. Çıktı: {json.dumps(output)[:2000]}")

    def _creation_len(item: tuple[str, dict]) -> int:
        return len(item[1].get("evm", {}).get("bytecode", {}).get("object", ""))

    # Birden fazla kontrat olabilir (kütüphane/yardımcı kontratlar); en
    # büyük creation bytecode'a sahip olanı ana kontrat kabul ediyoruz.
    key, contract = max(flat, key=_creation_len)

    creation_hex = contract.get("evm", {}).get("bytecode", {}).get("object", "")
    deployed_hex = contract.get("evm", {}).get("deployedBytecode", {}).get("object", "")
    abi = contract.get("abi", [])

    if not creation_hex:
        raise RuntimeError(f"'{key}' için creation bytecode boş.")

    creation_bytecode = bytes.fromhex(creation_hex)
    deployed_bytecode_size = len(bytes.fromhex(deployed_hex)) if deployed_hex else None
    exceeds_eip170 = deployed_bytecode_size is not None and deployed_bytecode_size > EIP170_MAX_DEPLOYED_BYTECODE_SIZE

    return {
        "contract_key": key,
        "abi": abi,
        "bytecode": creation_bytecode,
        "deployed_bytecode_size": deployed_bytecode_size,
        "exceeds_eip170": exceeds_eip170,
        "warnings": warnings,
    }


def compile_solidity(
    sol_path: Path,
    optimizer_runs: int = DEFAULT_OPTIMIZER_RUNS,
    via_ir: bool = True,
    evm_version: str = DEFAULT_EVM_VERSION,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """`sol_path`'i standard-JSON üzerinden derler, `{contract_key, abi,
    bytecode, deployed_bytecode_size, exceeds_eip170, warnings}` döner.

    `bytecode`: deploy için kullanılacak CREATION bytecode. EIP-170'in
    24576 byte sınırı DEPLOYED (runtime) bytecode'a uygulanır — ikisi
    ayrı ayrı raporlanır.
    """
    sol_path = Path(sol_path)
    source_code = sol_path.read_text(encoding="utf-8")
    standard_json = build_standard_json_input(sol_path.name, source_code, optimizer_runs, via_ir, evm_version)

    print(
        f"[solc] Derleniyor: {sol_path} (viaIR={via_ir}, optimizer_runs={optimizer_runs}, "
        f"evmVersion={evm_version}, timeout={timeout}s)"
    )

    try:
        result = subprocess.run(
            ["solc", "--standard-json"],
            input=json.dumps(standard_json),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=sol_path.parent,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", errors="replace")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(
            f"[solc] Derleme {timeout}s içinde bitmedi (viaIR açıkken uzayabilir). "
            f"'timeout' parametresini artırmayı dene.\nstdout={stdout[:2000]}\nstderr={stderr[:2000]}"
        ) from e

    if not result.stdout.strip():
        raise RuntimeError(f"[solc] Hiç çıktı üretmedi (returncode={result.returncode}). stderr:\n{result.stderr}")

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[solc] Çıktı JSON olarak parse edilemedi: {e}\nÇıktı:\n{result.stdout[:4000]}") from e

    parsed = _parse_standard_json_output(output)

    if parsed["warnings"]:
        print(f"[solc] {len(parsed['warnings'])} uyarı:")
        print(format_solc_messages(parsed["warnings"]))

    print(f"[solc] Ana kontrat: {parsed['contract_key']}")
    print(f"[solc] Creation bytecode: {len(parsed['bytecode'])} byte")
    if parsed["deployed_bytecode_size"] is not None:
        print(
            f"[solc] Deployed (runtime) bytecode: {parsed['deployed_bytecode_size']} byte "
            f"(EIP-170 sınırı: {EIP170_MAX_DEPLOYED_BYTECODE_SIZE})"
        )
        if parsed["exceeds_eip170"]:
            print(
                f"[solc] UYARI: deployed bytecode EIP-170 sınırını AŞIYOR "
                f"({parsed['deployed_bytecode_size']} > {EIP170_MAX_DEPLOYED_BYTECODE_SIZE}) — "
                f"bu kontrat gerçek bir zincirde DEPLOY EDİLEMEZ."
            )
    else:
        print("[solc] UYARI: deployedBytecode alınamadı, EIP-170 kontrolü atlanıyor.")

    return parsed


def _ensure_solc_version(version: str) -> None:
    """`solc-select` ile verilen sürümü kurar (zaten kuruluysa idempotent)
    ve aktive eder. `scripts/setup_colab.sh`'ın kurduğu `solc-select`
    aracına bağımlıdır."""
    print(f"[solc-select] '{version}' kuruluyor/aktive ediliyor...")

    install = subprocess.run(["solc-select", "install", version], capture_output=True, text=True)
    combined = (install.stdout + install.stderr).lower()
    if install.returncode != 0 and "already installed" not in combined and "already present" not in combined:
        raise RuntimeError(
            f"[solc-select] '{version}' kurulamadı (returncode={install.returncode}):\n"
            f"stdout={install.stdout}\nstderr={install.stderr}"
        )

    use = subprocess.run(["solc-select", "use", version], capture_output=True, text=True)
    if use.returncode != 0:
        raise RuntimeError(
            f"[solc-select] '{version}' aktive edilemedi (returncode={use.returncode}):\n"
            f"stdout={use.stdout}\nstderr={use.stderr}"
        )
    print(f"[solc-select] Aktif sürüm: {version}")


def compile_with_fallback_strategies(
    sol_path: Path,
    solc_versions: list[str] | None = None,
    strategies: list[dict] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Birden fazla (solc sürümü × via_ir × optimizer_runs) kombinasyonunu
    SIRAYLA dener, ilk BAŞARILI olanı `used_strategy` alanıyla birlikte
    döner. Her denemenin sonucu (başarılı/başarısız + hata özeti)
    loglanır. Hepsi başarısız olursa hepsinin özetini içeren TEK bir
    `RuntimeError` fırlatılır (hangi kombinasyonların denendiği ve neden
    başarısız olduğu kaybolmaz).

    ezkl'nin ürettiği Halo2Verifier.sol bazı ölçeklerde/sürümlerde
    "Stack too deep" ile derlenemeyebiliyor; hangi ayarın işe yarayacağı
    önceden güvenilir şekilde tahmin edilemez, bu yüzden gerçekten
    deneniyor.
    """
    sol_path = Path(sol_path)
    versions = solc_versions if solc_versions is not None else DEFAULT_SOLC_VERSIONS
    combos = strategies if strategies is not None else DEFAULT_STRATEGIES

    header = read_source_header(sol_path, 20)
    print(f"[solc] '{sol_path.name}' ilk 20 satır (pragma/sürüm bilgisi için):\n{header}")

    failed_attempts: list[str] = []
    current_version: str | None = None

    for version in versions:
        if version != current_version:
            try:
                _ensure_solc_version(version)
                current_version = version
            except RuntimeError as e:
                summary = f"solc-select({version}): {e}"
                print(f"[solc] BAŞARISIZ: {summary}")
                failed_attempts.append(summary)
                continue

        for combo in combos:
            label = f"solc={version}, viaIR={combo['via_ir']}, optimizer_runs={combo['optimizer_runs']}"
            print(f"[solc] Deneniyor: {label}")
            try:
                result = compile_solidity(
                    sol_path,
                    optimizer_runs=combo["optimizer_runs"],
                    via_ir=combo["via_ir"],
                    evm_version=combo.get("evm_version", DEFAULT_EVM_VERSION),
                    timeout=timeout,
                )
            except Exception as e:  # noqa: BLE001 - deneme sonucu kaydediliyor, tüm denemeler bitince tek hata fırlatılıyor
                summary = f"{label}: {type(e).__name__}: {e}"
                print(f"[solc] BAŞARISIZ: {summary}")
                failed_attempts.append(summary)
                continue

            print(f"[solc] BAŞARILI: {label}")
            result["used_strategy"] = label
            return result

    raise RuntimeError(
        f"[solc] {len(failed_attempts)} deneme de başarısız oldu:\n" + "\n".join(f"- {a}" for a in failed_attempts)
    )
