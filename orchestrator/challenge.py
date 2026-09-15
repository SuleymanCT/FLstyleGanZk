"""Faz D: `RoundManager.sol`'un ürettiği `challengeSeed`'den deterministik (z, c).

`fl.reference_utils.build_z_c`'yi (Faz C0'da yazılan, StyleGAN'a bağımlı
OLMAYAN saf fonksiyon) DOĞRUDAN yeniden kullanır — tek fark, tohumun
artık Colab'da seçilen bir int DEĞİL, zincirden gelen bir `bytes32`
`challengeSeed`'in türevi olması. Aynı `challengeSeed` HER ZAMAN aynı
(z, c) vermeli: protokolün adilliği (her site AYNI meydan okumayı
ispatlıyor) buna dayanıyor.

k=1 sabit — Faz C3'ün EIP-170 bulgusu gereği (k=4/k=8 verifier'ları
zincire deploy edilemiyor, bkz. `configs/circuit.yaml`/`docs/phase_c_report.md`).
"""

from __future__ import annotations

import torch

from fl.reference_utils import build_z_c

Z_DIM = 64
C_DIM = 5
CHALLENGE_K = 1

_SEED_MASK = (1 << 63) - 1  # torch.Generator.manual_seed'in kabul ettiği aralığa indirger


def challenge_seed_to_int(challenge_seed: bytes) -> int:
    """`bytes32` `challengeSeed`'i `torch.Generator.manual_seed`'in kabul
    ettiği aralığa (`0 <= seed < 2**63`) indirger. Deterministik: aynı
    bytes HER ZAMAN aynı int'i verir."""
    if not isinstance(challenge_seed, (bytes, bytearray)):
        raise TypeError(f"challenge_seed bytes/bytearray olmalı, alınan: {type(challenge_seed)}")
    if len(challenge_seed) == 0:
        raise ValueError("challenge_seed boş olamaz.")
    return int.from_bytes(bytes(challenge_seed), byteorder="big") & _SEED_MASK


def build_challenge_z_c(
    challenge_seed: bytes, k: int = CHALLENGE_K, z_dim: int = Z_DIM, c_dim: int = C_DIM
) -> tuple[torch.Tensor, torch.Tensor]:
    """`challenge_seed`'den deterministik `(z, c)` üretir — `k` varsayılan
    olarak `CHALLENGE_K=1` (Faz C3'ün EIP-170 kısıtı)."""
    seed_int = challenge_seed_to_int(challenge_seed)
    return build_z_c(k=k, z_dim=z_dim, c_dim=c_dim, seed=seed_int)
