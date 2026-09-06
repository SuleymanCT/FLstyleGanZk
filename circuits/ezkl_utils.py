"""ezkl'ye özgü ince yardımcılar.

BİLİNÇLİ TEST SINIRI: `ezkl` paketine bağımlıdır, gerçek çalışması
yalnızca Colab'da (ezkl kurulu) doğrulanabilir. Buradaki desen, aynı
kullanıcının `tez-projesi/src/zk.py` dosyasında `ezkl==23.0.5` ile
Colab'da KANITLANMIŞ event-loop sarmalayıcısıyla birebir aynıdır.
"""

from __future__ import annotations

import asyncio
import inspect


async def _get_srs_async(settings_path: str) -> None:
    """ezkl.get_srs çağrısının KENDİSİ (awaitable'ı oluşturma anı) zaten
    çalışan bir event loop içinde yapılmalı — pyo3-asyncio bağlaması
    çağrı anında aktif loop'a bağlanıyor. Aksi halde "RuntimeError: no
    running event loop" alınır (tez-projesi'nde Colab'da doğrulandı).
    """
    import ezkl

    result = ezkl.get_srs(settings_path)
    if inspect.isawaitable(result):
        await result


def run_get_srs(settings_path: str) -> None:
    """`_get_srs_async`'i bir event loop içinde çalıştırır."""
    try:
        asyncio.run(_get_srs_async(settings_path))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_get_srs_async(settings_path))
        finally:
            loop.close()
