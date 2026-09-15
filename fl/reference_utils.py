"""Faz C0: referans (z, c) girdileri üretme ve mapping çağrı imzasına
göre kwargs seçme.

StyleGAN'a bağımlı değildir (saf torch/inspect mantığı); yerelde tam
test edilebilir. Gerçek `G_ema.mapping` çağrısı ve pkl okuma
`scripts/make_reference_outputs.py`'de yaşar (BİLİNÇLİ TEST SINIRI —
orada test yok).
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F


def build_class_indices(k: int, c_dim: int) -> list[int]:
    if c_dim <= 0:
        raise ValueError(f"c_dim pozitif olmalı, alınan: {c_dim}")
    if k <= 0:
        raise ValueError(f"k pozitif olmalı, alınan: {k}")
    return [i % c_dim for i in range(k)]


def build_z_c(k: int, z_dim: int, c_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sabit tohumdan k adet z (k, z_dim) ve one-hot c (k, c_dim) üretir.

    Global `torch.manual_seed` state'ini KİRLETMEZ — yerel bir
    `torch.Generator` kullanır, böylece bu fonksiyonun çağrılması
    çağıran tarafın rastgelelik durumunu etkilemez.
    """
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(k, z_dim, generator=generator)
    indices = build_class_indices(k, c_dim)
    c = F.one_hot(torch.tensor(indices, dtype=torch.long), num_classes=c_dim).to(torch.float32)
    return z, c


def resolve_mapping_kwargs(
    signature: inspect.Signature,
    truncation_psi: float,
    truncation_cutoff,
) -> dict:
    """Verilen `mapping.forward` imzasında `truncation_psi`/`truncation_cutoff`
    parametre adları GERÇEKTEN varsa kwargs'a ekler, yoksa atlar ve neden
    atladığını loglar. z/c bu fonksiyonun kapsamı dışında — her zaman
    pozisyonel geçilir (StyleGAN2/3/XL'in tüm sürümlerinde ortak imza).
    """
    kwargs: dict = {}
    params = signature.parameters

    if "truncation_psi" in params:
        kwargs["truncation_psi"] = truncation_psi
    else:
        print(
            "[reference_utils] UYARI: mapping.forward imzasında 'truncation_psi' yok, "
            "gönderilmiyor. Bulunan parametreler: " + ", ".join(params.keys())
        )

    if "truncation_cutoff" in params:
        kwargs["truncation_cutoff"] = truncation_cutoff
    else:
        print(
            "[reference_utils] UYARI: mapping.forward imzasında 'truncation_cutoff' yok, "
            "gönderilmiyor. Bulunan parametreler: " + ", ".join(params.keys())
        )

    return kwargs
