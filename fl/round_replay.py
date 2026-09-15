"""Faz D: `round_runner.py`'nin 'yerel eğitim' adımının ince sarmalayıcısı.

CLAUDE.md madde 2/3 ("Base model yeniden eğitilmez. 15 round yeniden
eğitilmez." / "StyleGAN-XL eğitim döngüsünün matematiğine dokunulmaz.")
gereği bu fonksiyon GERÇEK EĞİTİM YAPMAZ. `orchestrator/round_runner.py`
canlı bir eğitim turu BAŞLATMIYOR — Faz A'da ZATEN üretilmiş, `raw_root`'ta
(salt okunur) duran site snapshot'larının REPLAY'i üzerinden protokolü
(challenge/ispat/agregasyon/finalize) simüle ediyor. Bu fonksiyon o
snapshot'ı yükler; canlı bir eğitim turu (Faz F'in kapsamı) ayrı bir iştir.

BİLİNÇLİ TEST SINIRI: gerçek bir `network-snapshot.pkl` ve StyleGAN-XL
reposu gerektirir (bkz. `fl/stylegan_xl_env.py`), yerelde test edilemez.
"""

from __future__ import annotations

from fl.stylegan_xl_env import load_network_pkl
from scripts.make_reference_outputs import resolve_site_pkl_path


def load_site_update(raw_root: str, stylegan_xl_repo: str, round_id: int, site_id: int) -> dict:
    """`(round_id, site_id)` için ZATEN eğitilmiş `G_ema.state_dict()`'i
    salt okunur `raw_root`'tan yükler (`load_network_pkl` ->
    `storage.pathguard.open_readonly` — yazma girişimi yapılmaz)."""
    pkl_path = resolve_site_pkl_path(raw_root, round_id, site_id)
    data = load_network_pkl(pkl_path, stylegan_xl_repo)
    if "G_ema" not in data:
        raise KeyError(f"'{pkl_path}' içinde 'G_ema' yok. Bulunan anahtarlar: {sorted(data.keys())}")
    return data["G_ema"].state_dict()
