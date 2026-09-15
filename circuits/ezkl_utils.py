"""ezkl'ye özgü ince yardımcılar.

BİLİNÇLİ TEST SINIRI: `run_async`/`run_get_srs` `ezkl` paketine
bağımlıdır, gerçek çalışması yalnızca Colab'da (ezkl kurulu)
doğrulanabilir. `parse_public_inputs` SAF bir fonksiyondur (sadece
`proof.json`'un içeriğini işler, ezkl'ye bağımlı değildir) —
`tests/test_ezkl_utils.py`'de sentetik `proof.json`'larla GERÇEKTEN
test edilir.
"""

from __future__ import annotations

import asyncio
import inspect

# BN254 (alt_bn128) skalar alanının modülüsü — ezkl'nin ürettiği
# `instances`/`public_inputs` değerleri Halo2/KZG devrelerinde bu
# alanın elemanlarıdır. Bir hex string'in little- ya da big-endian
# yorumunun HANGİSİNİN "makul" (gerçek bir alan elemanı) olduğunu
# ayırt etmek için kullanılıyor — varsaymak yerine DOĞRULAMAK için.
BN254_SCALAR_FIELD_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def run_async(fn, *args, **kwargs):
    """`fn(*args, **kwargs)`'i, ÇAĞRININ KENDİSİ bir event loop içinde
    olacak şekilde çalıştırır ve sonucunu döner.

    ezkl'nin bazı fonksiyonları (en az `get_srs` — tez-projesi'nde
    Colab'da doğrulandı — ve `create_evm_verifier` — Faz B'nin ilk
    Colab koşumunda ortaya çıktı) coroutine döndürür. Bu fonksiyonların
    awaitable'ı OLUŞTURULMA anı zaten çalışan bir event loop içinde
    olmalı (pyo3-asyncio bağlaması çağrı anında aktif loop'a
    bağlanıyor), aksi halde "RuntimeError: no running event loop"
    alınır. Bu yüzden `fn` burada, BİZİM açtığımız event loop'un
    İÇİNDE çağrılıyor — hazır bir coroutine/sonuç değil, çağrılacak
    fonksiyonun kendisi + argümanları alınıyor.

    Hangi ezkl fonksiyonunun async olduğu sürümden sürüme değişebilir
    (ve önceden tahmin edilmesi güvenilir değil — `create_evm_verifier`
    örneği bunu gösterdi). Bu yüzden burada senkron/async ayrımı
    ÖNCEDEN varsayılmıyor: `fn`'in sonucu `inspect.isawaitable` ile
    çalışma zamanında kontrol edilir, awaitable ise `await` edilir,
    değilse olduğu gibi kullanılır. Böylece ezkl'nin HANGİ fonksiyonu
    çağrılırsa çağrılsın (sync ya da async) aynı sarmalayıcı güvenle
    kullanılabilir.
    """

    async def _runner():
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    try:
        return asyncio.run(_runner())
    except RuntimeError:
        # asyncio.run() zaten çalışan bir loop varken çağrılırsa
        # ("cannot be called from a running event loop") buraya düşer;
        # ayrı bir loop açıp onunla tamamlıyoruz (tez-projesi'nde
        # Colab'da doğrulanmış geri dönüş deseni).
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_runner())
        finally:
            loop.close()


def run_get_srs(settings_path: str) -> None:
    import ezkl

    run_async(ezkl.get_srs, settings_path)


def _flatten(nested) -> list:
    """`instances` iç içe listeler halinde gelebilir (Faz D'nin gerçek
    Colab koşumunda gözlenen: tek elemanlı dış liste, 5 elemanlı iç
    liste) — derinliğe bakmadan tek düz bir listeye açar."""
    flat: list = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten(item))
        else:
            flat.append(item)
    return flat


def _hex_to_int(hex_str: str, byteorder: str) -> int:
    cleaned = hex_str[2:] if hex_str.lower().startswith("0x") else hex_str
    if len(cleaned) % 2 != 0:
        cleaned = "0" + cleaned  # tek sayıda hex karakter - baştan sıfırla tamamla
    return int.from_bytes(bytes.fromhex(cleaned), byteorder=byteorder)


def parse_public_inputs(proof_json: dict) -> list:
    """`proof.json`'un `instances`/`public_inputs` alanını `uint256[]`
    (Solidity `submitProof`'un beklediği) düz bir int listesine çevirir.

    Faz D'nin gerçek Colab koşumunda gözlenen biçim: iç içe liste, her
    eleman hex STRING — ve bu hex string'ler LITTLE-ENDIAN (`0x...06`
    biçiminde DEĞİL, `06000...0` biçiminde saklı; `int(x, 16)` ile
    düz okumak DEVASA, YANLIŞ bir sayı üretir). Endianness burada
    VARSAYILMIYOR: hem little hem big-endian yorumu hesaplanıp HANGİSİ
    BN254 skalar alanının (`BN254_SCALAR_FIELD_MODULUS`) İÇİNDE kalıyorsa
    o kullanılıyor, hangisinin seçildiği loglanıyor. İkisi de alan
    dışındaysa (şema tamamen beklenenden farklı) net bir `ValueError`
    ile durulur — sessizce yanlış bir sayı uydurulmaz.

    Zaten `int` olan elemanlar (bazı ezkl sürümleri/şemaları böyle
    dönebilir) olduğu gibi bırakılır — hiçbir dönüşüm uygulanmaz."""
    instances = proof_json.get("instances")
    if instances is None:
        instances = proof_json.get("public_inputs")
    if instances is None:
        raise KeyError(f"proof_json'da 'instances'/'public_inputs' anahtarı yok. Bulunan anahtarlar: {sorted(proof_json.keys())}")

    flat = _flatten(instances)
    if not flat:
        return []

    if all(isinstance(v, int) for v in flat):
        print("[parse_public_inputs] Elemanlar zaten int — dönüşüm uygulanmadı.")
        return list(flat)

    if not all(isinstance(v, str) for v in flat):
        raise TypeError(f"public_inputs karışık/beklenmeyen tipte elemanlar içeriyor: {[type(v).__name__ for v in flat][:10]}")

    little = [_hex_to_int(v, "little") for v in flat]
    big = [_hex_to_int(v, "big") for v in flat]
    little_valid = all(0 <= v < BN254_SCALAR_FIELD_MODULUS for v in little)
    big_valid = all(0 <= v < BN254_SCALAR_FIELD_MODULUS for v in big)

    if little_valid and not big_valid:
        print(f"[parse_public_inputs] little-endian yorumu geçerli (BN254 alanı içinde), kullanılıyor: {little}")
        return little
    if big_valid and not little_valid:
        print(f"[parse_public_inputs] big-endian yorumu geçerli (BN254 alanı içinde), kullanılıyor: {big}")
        return big
    if little_valid and big_valid:
        print(
            f"[parse_public_inputs] UYARI: hem little hem big-endian yorumu BN254 alanı içinde kalıyor, "
            f"ayırt edilemedi — little-endian TERCİH EDİLİYOR (Faz D'nin gerçek Colab koşumunda gözlenen biçim). "
            f"little={little} big={big}"
        )
        return little

    raise ValueError(
        f"Ne little-endian ne big-endian yorumu BN254 skalar alanının ({BN254_SCALAR_FIELD_MODULUS}) altında kalıyor — "
        f"proof.json şeması beklenenden farklı olabilir. little={little} big={big}"
    )
