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


def deterministic_unit_interval(challenge_seed: bytes, round_id: int, site) -> float:
    """`challenge_seed`+`round_id`+`site`'den `[0,1)` aralığında
    deterministik (kripto-rastgele DEĞİL, sadece tekrarlanabilir) bir
    değer üretir — sha256 tabanlı. Gerçek rastgelelik gerekmiyor: zincirdeki
    `challengeSeed` (`blockhash` türevi) zaten öngörülemez, buradaki
    hash sadece o öngörülemezliği (round,site) çiftine deterministik
    şekilde dağıtıyor.

    `site`: BİLEREK tip-esnek — bir `str` (GERÇEK bir EVM adresi, ör.
    `orchestrator/round_runner.py`'nin canlı akışının kullandığı) YA DA
    bir `int` (basit bir site indeksi, ör. `scripts/replay_proofs.py`'nin
    anvil hesap indeksleri 0-3) olabilir. İKİ çağıran tarafın FARKLI
    (ama kendi bağlamında DOĞRU) veri modelleri var: round_runner.py'de
    site'ların kanonik bir indeksi YOK, sadece kayıtlı cüzdan adresleri
    var; replay_proofs.py'de ise site kimliği anvil'in deterministik
    hesap indeksi — ikisini TEK bir temsile ZORLAMAK birini yapay/yanlış
    bir veri modeline sokardı. Bu yüzden `str(site).lower()` ile HER
    İKİSİ de TEK bir kanonik string'e indirgeniyor — `site=2` (int) ile
    `site="2"` (str) HER ZAMAN aynı deterministik değeri üretir
    (`str(2).lower() == str("2").lower() == "2"`), adresler için de
    büyük/küçük harf farkı elenir (`"0xABC".lower() == "0xabc"`).

    Faz E'nin gerçek Colab koşumunda `replay_proofs.py` `site`'ı bir
    `int` olarak geçirdi ama bu fonksiyon SADECE `str` (adres)
    varsayıyordu — `AttributeError: 'int' object has no attribute
    'lower'` ile çöktü. Artık YANLIŞ tipte sessizce çökmüyor."""
    if not isinstance(challenge_seed, (bytes, bytearray)):
        raise TypeError(f"challenge_seed bytes/bytearray olmalı, alınan: {type(challenge_seed)}")
    site_key = str(site).lower()
    payload = bytes(challenge_seed) + str(round_id).encode("utf-8") + site_key.encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") / 2**64


def must_prove(
    round_id: int,
    site,
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

    `site`: `str` (adres) ya da `int` (indeks) olabilir — bkz.
    `deterministic_unit_interval`'ın docstring'i (tip-esneklik gerekçesi).
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
    sites: list,
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
