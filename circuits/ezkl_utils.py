"""ezkl'ye özgü ince yardımcılar.

BİLİNÇLİ TEST SINIRI: `ezkl` paketine bağımlıdır, gerçek çalışması
yalnızca Colab'da (ezkl kurulu) doğrulanabilir.
"""

from __future__ import annotations

import asyncio
import inspect


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
