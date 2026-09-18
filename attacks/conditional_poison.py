"""Faz F saldırı 3 (EN ÖNEMLİSİ) — koşullu (sınıf-hedefli) zehirleme.

Mapping ağının sınıf gömme tablosunda (`mapping.embed.weight`) İKİ DR
sınıfının satırlarını TAKASLAR. Model GÖRSEL olarak normal (gerçekçi
fundus) görüntü üretmeye devam eder — `||ΔG||` neredeyse HİÇ değişmez
(tek bir 320-boyutlu satır çifti takas edildi, milyonlarca parametreden
sadece ikisi) — ama DR sınıfı `row_a` istendiğinde `row_b`'nin (ve
tersi) görsel karakteristiğini üretir. Bu yüzden:
  - `||ΔG|| > tau` norm kontrolü BÜYÜK OLASILIKLA bu saldırıyı KAÇIRIR
    (bkz. docs/phase_f_attacks.md — bu ölçülüp doğrulanacak, VARSAYILMAZ).
  - FID muhtemelen DÜŞMEZ (görüntüler hâlâ gerçekçi, sadece YANLIŞ
    sınıfa etiketli).
  - Asıl zarar `eval/metrics.py: class_confusion_matrix`'in ölçtüğü
    SINIF TUTARLILIĞINDA ortaya çıkar.

Saf state_dict operasyonudur — hiçbir model eğitilmez (CLAUDE.md
madde 2/3/6); ağırlık TAKASI bir eğitim adımı DEĞİLDİR.
"""

from __future__ import annotations

DEFAULT_EMBED_KEY = "mapping.embed.weight"


def swap_embed_rows(state_dict: dict, row_a: int, row_b: int, *, embed_key: str = DEFAULT_EMBED_KEY) -> dict:
    """`state_dict[embed_key]`'in `row_a`/`row_b` satırlarını takas eder,
    GERİ KALAN her şeyi (embed tablosunun diğer satırları DAHİL) AYNEN
    korur. `state_dict`'in kendisi (sığ kopya dışında) mutasyona
    UĞRATILMAZ — çağıranın orijinal dict'i hâlâ geçerli kalır."""
    if embed_key not in state_dict:
        raise KeyError(
            f"'{embed_key}' state_dict'te yok. Bulunan anahtarlardan ilk 10'u: "
            f"{sorted(state_dict.keys())[:10]}"
        )
    if row_a == row_b:
        raise ValueError(f"row_a ve row_b aynı ({row_a}) — takas anlamsız/etkisiz olur.")

    embed = state_dict[embed_key]
    if embed.dim() != 2:
        raise ValueError(f"'{embed_key}' 2 boyutlu bekleniyordu (num_classes, embed_dim), bulunan şekil: {tuple(embed.shape)}")
    num_rows = embed.shape[0]
    if not (0 <= row_a < num_rows) or not (0 <= row_b < num_rows):
        raise ValueError(f"row_a={row_a}/row_b={row_b}, embed tablosunun satır aralığının ([0,{num_rows})) DIŞINDA.")

    swapped = embed.clone()
    swapped[[row_a, row_b]] = swapped[[row_b, row_a]]

    result = dict(state_dict)
    result[embed_key] = swapped
    return result
