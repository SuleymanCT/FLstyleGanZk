"""StyleGAN-XL reposunu sys.path'e ekleyip pickle açan ortam katmanı.

BİLİNÇLİ TEST SINIRI: Bu modül gerçek bir autonomousvision/stylegan-xl
klonuna (dnnlib/, torch_utils/, legacy.py) ve gerçek bir
network-snapshot.pkl dosyasına ihtiyaç duyar. Yerel geliştirme
ortamında hiçbiri yok, bu yüzden bu modülün birim testi YOKTUR —
bu bilinçli bir sınırdır (PLAN.md Faz A notuna bakın), mock ile
doldurulmamıştır. Sadece Colab'da, gerçek veriyle doğrulanabilir.
"""

from __future__ import annotations

import os
import sys


def ensure_stylegan_xl_on_path(repo_path: str) -> None:
    print(f"[stylegan_xl_env] StyleGAN-XL repo kontrol ediliyor: {repo_path}")

    if not os.path.isdir(repo_path):
        raise RuntimeError(
            f"StyleGAN-XL reposu bulunamadı: '{repo_path}'.\n"
            f"Klonlamak için: git clone https://github.com/autonomousvision/stylegan-xl {repo_path}\n"
            f"Ya da --stylegan-xl-repo argümanıyla / configs/paths.yaml: stylegan_xl_repo ile "
            f"doğru yolu göster."
        )

    dnnlib_path = os.path.join(repo_path, "dnnlib")
    torch_utils_path = os.path.join(repo_path, "torch_utils")
    legacy_path = os.path.join(repo_path, "legacy.py")

    missing = [
        p
        for p in (("dnnlib/", dnnlib_path), ("torch_utils/", torch_utils_path), ("legacy.py", legacy_path))
        if not os.path.exists(p[1])
    ]
    if missing:
        missing_names = ", ".join(name for name, _ in missing)
        raise RuntimeError(
            f"'{repo_path}' bir StyleGAN-XL reposu gibi görünmüyor — eksik: {missing_names}. "
            f"Doğru repo (autonomousvision/stylegan-xl) klonlandığından emin ol."
        )

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        print(f"[stylegan_xl_env] '{repo_path}' sys.path'e eklendi.")
    else:
        print(f"[stylegan_xl_env] '{repo_path}' zaten sys.path'te.")

    try:
        import dnnlib  # noqa: F401
        import torch_utils  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"dnnlib/torch_utils importu başarısız oldu ({e}). "
            f"Repo path doğru görünüyor ama import edilemedi — "
            f"eksik bağımlılık (ör. torch sürümü uyuşmazlığı) olabilir."
        ) from e

    print("[stylegan_xl_env] dnnlib ve torch_utils başarıyla import edildi.")


def load_network_pkl(pkl_path: str, repo_path: str) -> dict:
    ensure_stylegan_xl_on_path(repo_path)

    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"pkl dosyası bulunamadı: {pkl_path}")

    import legacy  # StyleGAN-XL reposundan, ensure_stylegan_xl_on_path sonrası import edilebilir.

    print(f"[stylegan_xl_env] Yükleniyor: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = legacy.load_network_pkl(f)

    print(f"[stylegan_xl_env] Yüklendi. Anahtarlar: {sorted(data.keys())}")
    return data
