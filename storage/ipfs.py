"""Yerel kubo (go-ipfs) node'u için ince sarmalayıcı: `add(path) -> CID`,
`get(cid, path)`.

Zincire ağırlık verisi YAZILMAZ (bkz. CLAUDE.md/`contracts/RoundManager.sol`) —
büyük dosyalar (site güncellemeleri, agregasyon çıktıları) IPFS'e
yüklenip sadece CID'leri zincire yazılır. `kubo` CLI'sini (`ipfs` komutu)
bir subprocess olarak çağırır; node çalışmıyorsa (`ipfs id` başarısız)
NET bir hata verir, sessizce sahte bir CID uydurmaz (CLAUDE.md madde 6).

BİLİNÇLİ TEST SINIRI: gerçek bir yerel `ipfs daemon` gerektirir, yerelde
(bu oturumda kurulu değil) test edilemez — sadece Colab'da (`scripts/setup_colab.sh`
kubo'yu kurduktan sonra) gerçek koşumda doğrulanabilir. Saf CID
biçim-doğrulama (`is_valid_cid`) yerelde test edilir.
"""

from __future__ import annotations

import re
import subprocess

DEFAULT_TIMEOUT_SECONDS = 120.0

# CIDv0 (Qm... base58, 46 karakter) ve CIDv1 (b... base32) biçimlerinin
# kaba bir doğrulaması - kubo'nun ürettiği CID'lerin GERÇEKTEN bir CID
# gibi göründüğünü (boş string/hata mesajı DEĞİL) doğrulamak için.
_CIDV0_RE = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$")
_CIDV1_RE = re.compile(r"^b[a-z2-7]{20,}$")


def is_valid_cid(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_CIDV0_RE.match(value) or _CIDV1_RE.match(value))


def _run_ipfs(args: list, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    try:
        result = subprocess.run(["ipfs", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise RuntimeError(
            "[ipfs] 'ipfs' komutu bulunamadı — kubo kurulu değil. "
            "scripts/setup_colab.sh'ın kubo kurulumunu çalıştırdığından emin ol."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"[ipfs] 'ipfs {' '.join(args)}' {timeout}s içinde bitmedi.") from e

    if result.returncode != 0:
        raise RuntimeError(
            f"[ipfs] 'ipfs {' '.join(args)}' başarısız (returncode={result.returncode}). "
            f"stdout={result.stdout!r} stderr={result.stderr!r}. "
            f"Node çalışıyor mu? ('ipfs daemon' başlatılmış olmalı — bkz. scripts/setup_colab.sh)"
        )
    return result.stdout


def ensure_daemon_running(timeout: float = 10.0) -> None:
    """`ipfs id` ile node'un GERÇEKTEN ayakta olduğunu doğrular — daemon
    çalışmıyorsa (`ipfs add`/`ipfs get` zaten hata verir ama) bunu erken,
    net bir mesajla yapar."""
    _run_ipfs(["id"], timeout=timeout)


def add(path: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """`path`'i (dosya ya da dizin) yerel kubo node'una ekler, CID'i döner.
    `-Q` (quiet) ile sadece son CID satırı alınır (dizin eklerken kubo
    ara dosyaların CID'lerini de basar, `-Q` bunları bastırır)."""
    stdout = _run_ipfs(["add", "-r", "-Q", path], timeout=timeout)
    cid = stdout.strip()
    if not is_valid_cid(cid):
        raise RuntimeError(f"[ipfs] 'ipfs add {path}' geçerli bir CID döndürmedi: {cid!r}")
    return cid


def get(cid: str, output_path: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
    """`cid`'i `output_path`'e indirir (kubo'nun `-o` seçeneği)."""
    if not is_valid_cid(cid):
        raise ValueError(f"[ipfs] Geçersiz CID: {cid!r}")
    _run_ipfs(["get", cid, "-o", output_path], timeout=timeout)
