"""Faz F saldırı 1 — en kaba zehirleme: bir sitenin TÜM ağırlıklarını
aynı şekilli ama TAMAMEN rastgele (orijinal dağılımın istatistiklerini
KORUMAYAN, standart normal) tensörlerle değiştirir.

Saf state_dict operasyonudur — hiçbir model eğitilmez/yeniden eğitilmez
(CLAUDE.md madde 2/3/6). `scripts/run_attacks.py` bunu gerçek bir
site'ın `fl.round_replay.load_site_update` ile yüklenmiş TAM
`G_ema.state_dict()`'i üzerinde çağırır.
"""

from __future__ import annotations

import torch


def randomize_state_dict(state_dict: dict, *, generator: torch.Generator) -> dict:
    """Her float tensörü AYNI şekilde `torch.randn` ile doldurur —
    orijinal tensörün ölçeğini/istatistiklerini KORUMAZ, bilerek en
    kaba/naif saldırı budur. Float OLMAYAN tensörler (ör. bir sayaç
    buffer'ı — `num_batches_tracked` gibi) `randn` için anlamsız
    olduğundan DEĞİŞTİRİLMEZ, hangilerinin atlandığı açıkça loglanır
    (sessizce yutulmaz)."""
    result: dict = {}
    skipped: list[str] = []
    for key, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            result[key] = tensor.clone()
            skipped.append(key)
            continue
        result[key] = torch.randn(tuple(tensor.shape), generator=generator, dtype=torch.float32).to(tensor.dtype)

    if skipped:
        preview = skipped[:5]
        suffix = "..." if len(skipped) > 5 else ""
        print(f"[random_weights] {len(skipped)} float-olmayan tensör DEĞİŞTİRİLMEDİ: {preview}{suffix}")

    return result
