import asyncio

import pytest

from circuits.ezkl_utils import BN254_SCALAR_FIELD_MODULUS, parse_proof_bytes, parse_public_inputs, run_async


def test_run_async_with_sync_function():
    def add(a, b):
        return a + b

    assert run_async(add, 2, 3) == 5


def test_run_async_with_coroutine_function():
    async def add_async(a, b):
        await asyncio.sleep(0)
        return a + b

    assert run_async(add_async, 2, 3) == 5


def test_run_async_passes_kwargs():
    def f(a, b=0):
        return a - b

    assert run_async(f, 10, b=4) == 6


def test_run_async_propagates_exceptions():
    def boom():
        raise ValueError("kaboom")

    try:
        run_async(boom)
    except ValueError as e:
        assert "kaboom" in str(e)
    else:
        raise AssertionError("ValueError bekleniyordu")


def _little_hex(value: int, num_bytes: int = 32) -> str:
    return value.to_bytes(num_bytes, "little").hex()


def _big_hex(value: int, num_bytes: int = 32) -> str:
    return value.to_bytes(num_bytes, "big").hex()


def test_parse_public_inputs_detects_little_endian_correctly():
    # ezkl'nin Faz D'nin gerçek Colab koşumunda gözlenen biçimi:
    # little-endian saklanmış hex string (int(x, 16) ile okunursa devasa,
    # yanlış bir sayı üretir).
    assert parse_public_inputs({"instances": [[_little_hex(6)]]}) == [6]


def test_parse_public_inputs_detects_big_endian_when_that_is_the_valid_one():
    # 200: son bayta sığıyor ama little-endian yanlış okuması (200 * 2^248)
    # BN254 alanının DIŞINA taşıyor - bu yüzden big-endian yorumu TEK
    # geçerli seçenek olur (42 gibi küçük değerlerde HER İKİ yorum da
    # yanlışlıkla alan içinde kalabiliyor, bu test o belirsizliği önlüyor).
    assert parse_public_inputs({"instances": [[_big_hex(200)]]}) == [200]


def test_parse_public_inputs_flattens_nested_lists():
    values = [1, 2, 3, 4, 5]
    result = parse_public_inputs({"instances": [[_little_hex(v) for v in values]]})
    assert result == values


def test_parse_public_inputs_handles_flat_list_not_nested():
    values = [7, 8]
    result = parse_public_inputs({"public_inputs": [_little_hex(v) for v in values]})
    assert result == values


def test_parse_public_inputs_passes_through_already_int_values():
    assert parse_public_inputs({"instances": [[1, 2, 3]]}) == [1, 2, 3]


def test_parse_public_inputs_real_colab_shape_five_little_endian_instances():
    # Faz D'nin gerçek Colab koşumunda gözlenen şekil: tek elemanlı dış
    # liste, 5 elemanlı iç liste, hepsi little-endian hex string.
    field_element = BN254_SCALAR_FIELD_MODULUS - 12345  # gerçekçi büyüklükte, alan içi bir değer
    values = [0, field_element, 1, 1, 6]
    result = parse_public_inputs({"instances": [[_little_hex(v) for v in values]]})
    assert result == values
    assert all(0 <= v < BN254_SCALAR_FIELD_MODULUS for v in result)


def test_parse_public_inputs_raises_when_neither_endianness_is_valid():
    # 40 baytlık (32'den uzun) bir değer - ne little ne big-endian yorumu
    # BN254 alanının altında kalır.
    garbage = "ff" * 40
    with pytest.raises(ValueError, match="skalar alanının"):
        parse_public_inputs({"instances": [[garbage]]})


def test_parse_public_inputs_raises_on_missing_keys():
    with pytest.raises(KeyError, match="instances"):
        parse_public_inputs({"proof": "0xdead"})


def test_parse_public_inputs_raises_on_mixed_types():
    with pytest.raises(TypeError):
        parse_public_inputs({"instances": [[1, "not-consistent"]]})


def test_parse_public_inputs_empty_instances_returns_empty_list():
    assert parse_public_inputs({"instances": []}) == []


def test_parse_proof_bytes_from_int_list():
    # Faz D'nin gerçek Colab koşumunda gözlenen biçim: proof bir INT
    # LİSTESİ (hex string DEĞİL).
    values = [16, 209, 217, 7, 255, 0]
    assert parse_proof_bytes({"proof": values}) == bytes(values)


def test_parse_proof_bytes_from_hex_string_without_prefix():
    raw = bytes([1, 2, 3, 254, 255])
    assert parse_proof_bytes({"proof": raw.hex()}) == raw


def test_parse_proof_bytes_from_hex_string_with_0x_prefix():
    raw = bytes([10, 20, 30])
    assert parse_proof_bytes({"proof": "0x" + raw.hex()}) == raw


def test_parse_proof_bytes_from_already_bytes():
    raw = bytes([1, 2, 3])
    assert parse_proof_bytes({"proof": raw}) == raw


def test_parse_proof_bytes_from_bytearray():
    raw = bytearray([9, 9, 9])
    result = parse_proof_bytes({"proof": raw})
    assert isinstance(result, bytes)
    assert result == bytes(raw)


def test_parse_proof_bytes_raises_on_missing_key():
    with pytest.raises(KeyError, match="proof"):
        parse_proof_bytes({"instances": []})


def test_parse_proof_bytes_raises_on_odd_length_hex_string():
    with pytest.raises(ValueError, match="tek sayıda"):
        parse_proof_bytes({"proof": "abc"})


def test_parse_proof_bytes_raises_on_out_of_range_int_list():
    with pytest.raises(ValueError, match="0-255"):
        parse_proof_bytes({"proof": [16, 300, -1]})


def test_parse_proof_bytes_raises_on_unrecognized_type():
    with pytest.raises(TypeError, match="beklenmeyen tipte"):
        parse_proof_bytes({"proof": 12345})
