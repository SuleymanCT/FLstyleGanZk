"""Korumalı (salt okunur) Drive köklerine yazmayı kod seviyesinde engelleyen bekçi.

"parallel_final'a asla yazma" kuralı artık sadece CLAUDE.md'de bir cümle
değil — her yazma çağrısından önce `assert_writable` çağrılmalı (bkz.
CLAUDE.md). Bir yol PROTECTED_MARKERS'tan birini içeriyorsa yazma
girişimi burada, dosya sistemine dokunmadan, RuntimeError ile durur.
"""

from __future__ import annotations

import os

PROTECTED_MARKERS = ["Generative_Image", "FL_Experiments", "parallel_final", "StyleGANTrain"]


def assert_writable(path: str) -> None:
    normalized = os.path.abspath(path).replace("\\", "/")
    for marker in PROTECTED_MARKERS:
        if marker in normalized:
            raise RuntimeError(
                f"YAZMA ENGELLENDİ: '{path}' korumalı bir işaretçi içeriyor ('{marker}'). "
                f"{PROTECTED_MARKERS} altındaki hiçbir yola bu repodan yazılamaz/silinemez — "
                f"bunlar salt okunur Drive kökleridir. Çıktıyı zk_root/results_dir gibi "
                f"ayrı bir konuma yönlendir."
            )


def open_readonly(path: str):
    return open(path, "rb")
