"""anvil (Foundry'nin yerel EVM node'u) başlatma/kapatma ve çıktısını ayrıştırma.

Faz B'nin oyuncak zinciri VE Faz D'nin `tests/test_contracts.py`'ı (gerçek
ezkl ispatlarıyla) tarafından ortak kullanılır. Saf metin ayrıştırma
fonksiyonları (`parse_anvil_accounts`, `is_anvil_ready`, `find_free_port`)
gerçek bir anvil süreci gerektirmez, sentetik metinle test edilir.
`AnvilProcess` (gerçek subprocess başlatan kısım) TEST EDİLMEZ — anvil
yerelde yok, sadece Colab'da gerçek koşumda doğrulanabilir.
"""

from __future__ import annotations

import re
import socket
import subprocess
import time

# Anvil'in varsayılan (mnemonic "test test test test test test test test
# test test test junk") ilk hesabı — kamuya açık, belgelenmiş, HER anvil
# kurulumunda aynı sabit değer (özel bir ayar/tohum verilmediği sürece).
# Çıktı ayrıştırma başarısız olursa buna düşülür (uydurma değil).
DEFAULT_ANVIL_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
DEFAULT_ANVIL_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

_ADDRESS_LINE = re.compile(r"^\(\d+\)\s+(0x[0-9a-fA-F]{40})", re.MULTILINE)
_KEY_LINE = re.compile(r"^\(\d+\)\s+(0x[0-9a-fA-F]{64})", re.MULTILINE)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_anvil_ready(stdout_text: str) -> bool:
    return "Listening on" in stdout_text


def parse_anvil_accounts(stdout_text: str) -> list[tuple[str, str]]:
    """anvil'in başlangıç bannerındaki "Available Accounts"/"Private Keys"
    bloklarını ayrıştırıp (adres, private_key) listesi döner (index'e göre
    eşleştirilir). Ayrıştırılamazsa boş liste döner (uydurmaz)."""
    addresses = _ADDRESS_LINE.findall(stdout_text)
    keys = _KEY_LINE.findall(stdout_text)
    return list(zip(addresses, keys))


class AnvilProcess:
    """`with AnvilProcess() as anvil: ...` — anvil'i başlatır, hazır olana
    kadar bekler, çıkışta güvenle kapatır."""

    def __init__(self, ready_timeout: float = 30.0):
        self.ready_timeout = ready_timeout
        self.port: int | None = None
        self.proc: subprocess.Popen | None = None
        self.accounts: list[tuple[str, str]] = []
        self._stdout_lines: list[str] = []

    @property
    def rpc_url(self) -> str:
        if self.port is None:
            raise RuntimeError("AnvilProcess henüz başlatılmadı.")
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "AnvilProcess":
        self.port = find_free_port()
        print(f"[anvil] Başlatılıyor: port={self.port}")
        self.proc = subprocess.Popen(
            ["anvil", "--port", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        deadline = time.monotonic() + self.ready_timeout
        buffer = ""
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"[anvil] Süreç beklenmedik şekilde sonlandı (returncode={self.proc.returncode}). "
                    f"Çıktı:\n{buffer}"
                )
            line = self.proc.stdout.readline()
            if not line:
                continue
            buffer += line
            self._stdout_lines.append(line)
            if is_anvil_ready(buffer):
                self.accounts = parse_anvil_accounts(buffer)
                if not self.accounts:
                    print(
                        "[anvil] UYARI: hesap listesi çıktıdan ayrıştırılamadı, "
                        "Anvil'in belgelenmiş varsayılan ilk hesabına düşülüyor."
                    )
                    self.accounts = [(DEFAULT_ANVIL_ADDRESS, DEFAULT_ANVIL_PRIVATE_KEY)]
                print(f"[anvil] Hazır: {self.rpc_url} ({len(self.accounts)} hesap)")
                return self

        self.__exit__(None, None, None)
        raise RuntimeError(
            f"[anvil] {self.ready_timeout}s içinde hazır olmadı ('Listening on' görülmedi). "
            f"Çıktı:\n{buffer}"
        )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.proc is None:
            return
        print("[anvil] Kapatılıyor...")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("[anvil] terminate() zaman aşımına uğradı, kill() deneniyor.")
            self.proc.kill()
            self.proc.wait(timeout=10)
        print("[anvil] Kapatıldı.")
