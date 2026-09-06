from chain.anvil import find_free_port, is_anvil_ready, parse_anvil_accounts

SAMPLE_ANVIL_OUTPUT = """
                             _   _
                            (_) | |
      __ _   _ __   __   __  _  | |
     / _` | | '_ \\  \\ \\ / / | | | |
    | (_| | | | | |  \\ V /  | | | |
     \\__,_| |_| |_|   \\_/   |_| |_|

    0.2.0

Available Accounts
==================

(0) 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 (10000.000000000000000000 ETH)
(1) 0x70997970C51812dc3A010C7d01b50e0d17dc79C8 (10000.000000000000000000 ETH)

Private Keys
==================

(0) 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
(1) 0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d

Listening on 127.0.0.1:8545
"""


def test_find_free_port_returns_usable_port():
    port = find_free_port()
    assert isinstance(port, int)
    assert 0 < port < 65536


def test_find_free_port_two_calls_can_bind_independently():
    import socket

    port = find_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        # ikinci bir çağrı farklı bir port bulmalı (ilki hâlâ dinlemede)
        other = find_free_port()
        assert other != port


def test_is_anvil_ready_detects_listening_line():
    assert is_anvil_ready(SAMPLE_ANVIL_OUTPUT) is True
    assert is_anvil_ready("henuz hazir degil") is False


def test_parse_anvil_accounts_pairs_addresses_with_keys():
    accounts = parse_anvil_accounts(SAMPLE_ANVIL_OUTPUT)
    assert accounts == [
        ("0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266", "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"),
        ("0x70997970C51812dc3A010C7d01b50e0d17dc79C8", "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"),
    ]


def test_parse_anvil_accounts_returns_empty_on_unrecognized_text():
    assert parse_anvil_accounts("beklenmeyen cikti, hesap yok") == []
