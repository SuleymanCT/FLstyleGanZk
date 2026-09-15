"""İki tensörü karşılaştırma istatistikleri — saf, StyleGAN'a bağımlı değil.

`scripts/verify_rebuild.py` (Faz C1) ve ileride Faz C4'ün (kuantizasyon
etkisi: w_q vs fp32 w) ortak kullanacağı yer.
"""

from __future__ import annotations

import torch


def compute_comparison_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"Şekiller uyuşmuyor: {tuple(a.shape)} vs {tuple(b.shape)}")

    a_flat = a.detach().to(torch.float32).flatten()
    b_flat = b.detach().to(torch.float32).flatten()

    diff = (a_flat - b_flat).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    denom = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    cosine_similarity = (torch.dot(a_flat, b_flat) / denom).item() if denom > 0 else None

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_similarity": cosine_similarity,
    }


def assert_num_ws_copies_identical(w: torch.Tensor) -> None:
    """Referans `w`'nin (k, num_ws, w_dim) şeklinde num_ws boyunca HEPSİ
    birebir aynı olmalı (StyleGAN'ın broadcast/repeat'inden geldiği
    için — bkz. circuits/rebuild_mapping.py docstring'i). Değilse
    referans dosyası beklenenden farklı bir yapıda demektir — net hata.
    """
    if w.dim() != 3:
        raise RuntimeError(f"'w' 3 boyutlu (k, num_ws, w_dim) bekleniyordu, bulunan şekil: {tuple(w.shape)}")
    reference_slice = w[:, 0:1, :]
    if not torch.allclose(w, reference_slice.expand_as(w)):
        max_diff = (w - reference_slice.expand_as(w)).abs().max().item()
        raise RuntimeError(
            f"'w'nin num_ws kopyaları BİREBİR AYNI değil (max fark={max_diff}). "
            f"Bu beklenmiyordu (broadcast salt tekrar olmalıydı) — referans dosyasını/varsayımları gözden geçir."
        )
