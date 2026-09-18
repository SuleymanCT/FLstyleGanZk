"""Faz F saldırı 2 — ölçeklenmiş ΔG zehirlemesi.

`ΔG = site_ağırlığı - önceki_global`; zehirlenmiş güncelleme
`önceki_global + scale * ΔG` olarak üretilir. `scale=1` site'ın
GERÇEK güncellemesiyle birebir aynıdır (saldırı yok — bu bir
doğrulama/sağlaması noktası); `scale=10/50/100` FedAvg'ın normalde
gördüğü ΔG büyüklüğünün çok üzerinde bir güncelleme üretir.

Saf state_dict operasyonudur — hiçbir model eğitilmez (CLAUDE.md
madde 2/3/6).
"""

from __future__ import annotations

import torch

from fl.fedavg_utils import assert_matching_keys

DEFAULT_SCALES = (10.0, 50.0, 100.0)


def scale_delta(site_state: dict, prev_global_state: dict, scale: float) -> dict:
    """`prev_global_state + scale * (site_state - prev_global_state)`,
    anahtar/tensör bazında, hep fp32 aritmetiğiyle (CLAUDE.md madde 5 —
    kuantizasyon burada YOK), sonuç orijinal `site_state` tensörünün
    dtype'ına geri çevrilir. Anahtar kümeleri BİREBİR eşleşmiyorsa
    (`assert_matching_keys`) sessizce kesişim ALINMAZ, net hata verilir."""
    assert_matching_keys(set(site_state.keys()), set(prev_global_state.keys()), context="scale_delta")

    result: dict = {}
    for key, site_tensor in site_state.items():
        site_fp32 = site_tensor.detach().cpu().to(torch.float32)
        prev_fp32 = prev_global_state[key].detach().cpu().to(torch.float32)
        delta = site_fp32 - prev_fp32
        result[key] = (prev_fp32 + scale * delta).to(site_tensor.dtype)
    return result
