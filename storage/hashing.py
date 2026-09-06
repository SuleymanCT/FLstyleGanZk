"""Yerel ve Colab arasında birebir aynı sonucu veren canonical hash.

Federe turların taahhütleri (on-chain'e yazılan globalHash/commitment)
bu fonksiyona dayanır: aynı ağırlıklar, hangi ortamda/hangi anahtar
sırasıyla dict'e konmuş olursa olsun, aynı hash'i üretmeli.
"""

import hashlib

import numpy as np
import torch


def canonical_hash(state_dict: dict) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        array = tensor.detach().cpu().to(torch.float32).numpy()
        hasher.update(array.astype("<f4").tobytes())
    return hasher.hexdigest()
