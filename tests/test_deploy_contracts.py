from scripts.deploy_contracts import build_deployment_record


def test_build_deployment_record_contains_all_fields():
    record = build_deployment_record("0xRoundManager", [{"type": "function"}], "http://127.0.0.1:8545")
    assert record["round_manager_address"] == "0xRoundManager"
    assert record["round_manager_abi"] == [{"type": "function"}]
    assert record["rpc_url"] == "http://127.0.0.1:8545"
    assert "deployed_at" in record
