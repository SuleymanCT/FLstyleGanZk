import pytest

from chain.solc import (
    EIP170_MAX_DEPLOYED_BYTECODE_SIZE,
    _parse_standard_json_output,
    build_standard_json_input,
    format_solc_messages,
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
