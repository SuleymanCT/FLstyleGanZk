"""configs/paths.yaml yükleyici.

Hiçbir script'te yol gömülmez; bunun yerine bu modül `TEZ_ENV`'e göre
`local` veya `colab` bloğunu döndürür, CLI argümanları bu değerleri
override edebilir.
"""

from __future__ import annotations

import os

import yaml

VALID_ENVS = ("local", "colab")


def load_paths(env: str | None = None, config_path: str = "configs/paths.yaml") -> dict:
    resolved_env = env or os.environ.get("TEZ_ENV", "local")
    if resolved_env not in VALID_ENVS:
        raise ValueError(
            f"Bilinmeyen ortam '{resolved_env}'. Beklenen: {VALID_ENVS}. "
            f"(TEZ_ENV ortam değişkenini veya --env argümanını kontrol et.)"
        )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if resolved_env not in data:
        raise ValueError(
            f"'{config_path}' içinde '{resolved_env}' bloğu yok. "
            f"Bulunan bloklar: {sorted(data.keys())}"
        )

    return data[resolved_env]
