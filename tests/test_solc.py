import pytest

import chain.solc as solc_module
from chain.solc import (
    EIP170_MAX_DEPLOYED_BYTECODE_SIZE,
    _parse_standard_json_output,
    build_standard_json_input,
    compile_with_fallback_strategies,
    format_solc_messages,
    read_source_header,
)


def _fake_output(bin_hex: str, deployed_hex: str, abi=None, errors=None):
    return {
        "errors": errors or [],
        "contracts": {
            "Verifier.sol": {
                "Verifier": {
                    "abi": abi or [],
                    "evm": {
                        "bytecode": {"object": bin_hex},
                        "deployedBytecode": {"object": deployed_hex},
                    },
                }
            }
        },
    }


def test_build_standard_json_input_has_expected_settings():
    payload = build_standard_json_input("Verifier.sol", "contract Verifier {}", optimizer_runs=200, via_ir=True, evm_version="shanghai")
    assert payload["language"] == "Solidity"
    assert payload["sources"]["Verifier.sol"]["content"] == "contract Verifier {}"
    assert payload["settings"]["viaIR"] is True
    assert payload["settings"]["optimizer"] == {"enabled": True, "runs": 200}
    assert payload["settings"]["evmVersion"] == "shanghai"


def test_parse_standard_json_output_extracts_bytecode_and_abi():
    output = _fake_output("6001", "6002", abi=[{"type": "function", "name": "verify"}])
    result = _parse_standard_json_output(output)

    assert result["bytecode"] == bytes.fromhex("6001")
    assert result["deployed_bytecode_size"] == 2
    assert result["exceeds_eip170"] is False
    assert result["abi"] == [{"type": "function", "name": "verify"}]
    assert result["contract_key"] == "Verifier.sol:Verifier"


def test_parse_standard_json_output_flags_eip170_violation():
    huge_deployed = "00" * (EIP170_MAX_DEPLOYED_BYTECODE_SIZE + 1)
    output = _fake_output("6001", huge_deployed)
    result = _parse_standard_json_output(output)

    assert result["deployed_bytecode_size"] == EIP170_MAX_DEPLOYED_BYTECODE_SIZE + 1
    assert result["exceeds_eip170"] is True


def test_parse_standard_json_output_raises_on_fatal_error():
    output = {
        "errors": [{"severity": "error", "formattedMessage": "Stack too deep"}],
        "contracts": {},
    }
    with pytest.raises(RuntimeError, match="Stack too deep"):
        _parse_standard_json_output(output)


def test_parse_standard_json_output_does_not_raise_on_warnings_only():
    output = _fake_output("6001", "6002", errors=[{"severity": "warning", "formattedMessage": "unused variable"}])
    result = _parse_standard_json_output(output)
    assert len(result["warnings"]) == 1
    assert result["bytecode"] == bytes.fromhex("6001")


def test_parse_standard_json_output_picks_largest_creation_bytecode_among_multiple_contracts():
    output = {
        "errors": [],
        "contracts": {
            "Verifier.sol": {
                "Helper": {"abi": [], "evm": {"bytecode": {"object": "6001"}, "deployedBytecode": {"object": "6001"}}},
                "Verifier": {"abi": [], "evm": {"bytecode": {"object": "600160026003"}, "deployedBytecode": {"object": "6001"}}},
            }
        },
    }
    result = _parse_standard_json_output(output)
    assert result["contract_key"] == "Verifier.sol:Verifier"


def test_format_solc_messages_falls_back_to_message_field():
    messages = format_solc_messages([{"message": "plain message"}])
    assert "plain message" in messages


def test_read_source_header_returns_first_n_lines(tmp_path):
    sol_path = tmp_path / "Verifier.sol"
    sol_path.write_text("\n".join(f"line{i}" for i in range(30)), encoding="utf-8")

    header = read_source_header(sol_path, num_lines=5)

    assert header.count("\n") == 5  # ilk 5 satırın hepsi \n ile bitiyor
    assert "line0" in header
    assert "line4" in header
    assert "line5" not in header


def test_read_source_header_handles_short_file(tmp_path):
    sol_path = tmp_path / "Verifier.sol"
    sol_path.write_text("pragma solidity ^0.8.20;\n", encoding="utf-8")

    header = read_source_header(sol_path, num_lines=20)

    assert header == "pragma solidity ^0.8.20;\n"


def test_compile_with_fallback_strategies_returns_first_success(tmp_path, monkeypatch):
    sol_path = tmp_path / "Verifier.sol"
    sol_path.write_text("pragma solidity ^0.8.20;\ncontract Verifier {}\n", encoding="utf-8")

    monkeypatch.setattr(solc_module, "_ensure_solc_version", lambda version: None)

    attempted = []

    def fake_compile(path, optimizer_runs, via_ir, evm_version, timeout):
        attempted.append((optimizer_runs, via_ir))
        if len(attempted) < 3:
            raise RuntimeError("Stack too deep")
        return {"bytecode": b"\x60\x01", "deployed_bytecode_size": 1, "exceeds_eip170": False, "abi": [], "warnings": []}

    monkeypatch.setattr(solc_module, "compile_solidity", fake_compile)

    result = compile_with_fallback_strategies(sol_path, solc_versions=["0.8.20"])

    assert len(attempted) == 3  # ilk iki kombinasyon basarisiz, ucuncu basarili
    assert "used_strategy" in result
    assert result["bytecode"] == b"\x60\x01"


def test_compile_with_fallback_strategies_raises_combined_error_when_all_fail(tmp_path, monkeypatch):
    sol_path = tmp_path / "Verifier.sol"
    sol_path.write_text("pragma solidity ^0.8.20;\ncontract Verifier {}\n", encoding="utf-8")

    monkeypatch.setattr(solc_module, "_ensure_solc_version", lambda version: None)

    def always_fail(path, optimizer_runs, via_ir, evm_version, timeout):
        raise RuntimeError(f"Stack too deep (runs={optimizer_runs}, via_ir={via_ir})")

    monkeypatch.setattr(solc_module, "compile_solidity", always_fail)

    with pytest.raises(RuntimeError) as exc_info:
        compile_with_fallback_strategies(
            sol_path,
            solc_versions=["0.8.20"],
            strategies=[{"via_ir": True, "optimizer_runs": 1}, {"via_ir": False, "optimizer_runs": 200}],
        )

    message = str(exc_info.value)
    assert "2 deneme" in message
    assert "runs=1" in message
    assert "runs=200" in message
