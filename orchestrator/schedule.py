"""Faz D: ispat takvimi — hangi (round, site) çiftinin ispat üretmesi
GEREKTİĞİNİ, hiçbir zincir/ezkl bağımlılığı olmadan SAF mantıkla belirler.
`round_runner.py` bunu çağırıp sonucuna göre ezkl ispatı üretir/üretmez.
"""

from __future__ import annotations

import hashlib

import yaml

VALID_MODES = ("sampled", "full")


def load_schedule_config(config_path: str = "configs/schedule.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    required = (
        "proof_schedule",
        "reputation_initial",
        "reputation_penalty",
        "reputation_bonus",
        "tau_norm_threshold",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"'{config_path}' içinde eksik anahtarlar: {missing}")
    return data


def deterministic_unit_interval(challenge_seed: bytes, round_id: int, site: str) -> float:
    """`challenge_seed`+`round_id`+`site`'den `[0,1)` aralığında
    deterministik (kripto-rastgele DEĞİL, sadece tekrarlanabilir) bir
    değer üretir — sha256 tabanlı. Gerçek rastgelelik gerekmiyor: zincirdeki
    `challengeSeed` (`blockhash` türevi) zaten öngörülemez, buradaki
    hash sadece o öngörülemezliği (round,site) çiftine deterministik
    şekilde dağıtıyor."""
    if not isinstance(challenge_seed, (bytes, bytearray)):
        raise TypeError(f"challenge_seed bytes/bytearray olmalı, alınan: {type(challenge_seed)}")
    payload = bytes(challenge_seed) + str(round_id).encode("utf-8") + site.lower().encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") / 2**64


def must_prove(
    round_id: int,
    site: str,
    *,
    challenge_seed: bytes,
    reputation: int,
    reputation_threshold: int,
    fixed_rounds: list[int],
    random_ratio: float | None,
    mode: str = "sampled",
) -> bool:
    """`mode="full"`: HER (round,site) ispat üretir (makalenin
    karşılaştırma tabanı — örnekleme YOKKEN toplam maliyet).
    `mode="sampled"` (varsayılan, üretim modu) üç kuralı OR'lar:
      1. `round_id` `fixed_rounds` içindeyse -> True (round genelinde ZORUNLU).
      2. `reputation < reputation_threshold` -> True (güvenilmez site HER ZAMAN izlenir).
      3. Aksi halde `deterministic_unit_interval(...) < random_ratio` -> True.
    """
    if mode == "full":
        return True
    if mode != "sampled":
        raise ValueError(f"Bilinmeyen mod: {mode!r} (geçerli: {VALID_MODES})")

    if round_id in fixed_rounds:
        return True
    if reputation < reputation_threshold:
        return True
    if not random_ratio:
        return False
    return deterministic_unit_interval(challenge_seed, round_id, site) < random_ratio


def build_round_schedule(
    round_id: int,
    sites: list[str],
    *,
    challenge_seed: bytes,
    reputations: dict,
    config: dict,
    mode: str = "sampled",
) -> dict:
    """Bir round için TÜM site'ların ispat gerekip gerekmediğini toplu
    hesaplar — `round_runner.py`'nin tek çağrı noktası."""
    proof_schedule = config["proof_schedule"]
    return {
        site: must_prove(
            round_id,
            site,
            challenge_seed=challenge_seed,
            reputation=reputations.get(site, config["reputation_initial"]),
            reputation_threshold=proof_schedule["reputation_threshold"],
            fixed_rounds=proof_schedule["fixed_rounds"],
            random_ratio=proof_schedule["random_ratio"],
            mode=mode,
        )
        for site in sites
    }
