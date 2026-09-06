"""Tam bir G_ema state_dict'inden mapping/embedding shard'ı çıkarma.

Bu modül torch tensörleri dışında hiçbir StyleGAN bağımlılığı içermez;
prefix filtreleme ve meta üretimi saf sözlük/tensor işlemleridir.
"""

from __future__ import annotations

from storage.hashing import canonical_hash


def extract_prefixed_state_dict(full_state_dict: dict, prefixes: list[str]) -> dict:
    shard = {
        key: tensor
        for key, tensor in full_state_dict.items()
        if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)
    }
    if not shard:
        raise ValueError(
            f"Hiçbir anahtar prefixlerle eşleşmedi: {prefixes}. "
            f"Mevcut anahtarlardan örnek: {list(full_state_dict.keys())[:10]}"
        )
    return shard


def build_shard_meta(full_state_dict: dict, shard_state_dict: dict, extra: dict | None = None) -> dict:
    full_param_count = sum(t.numel() for t in full_state_dict.values())
    shard_param_count = sum(t.numel() for t in shard_state_dict.values())

    meta = {
        "full_generator_hash": canonical_hash(full_state_dict),
        "full_param_count": full_param_count,
        "shard_param_count": shard_param_count,
        "shard_keys": sorted(shard_state_dict.keys()),
        "shard_shapes": {k: list(v.shape) for k, v in shard_state_dict.items()},
    }
    if extra:
        meta.update(extra)
    return meta
