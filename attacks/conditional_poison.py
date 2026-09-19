"""Faz F saldırı 3 (EN ÖNEMLİSİ) — koşullu (sınıf-hedefli) zehirleme.

Mapping ağının sınıf gömme tablosunda (`mapping.embed.weight`) İKİ DR
sınıfının satırlarını TAKASLAR. Model GÖRSEL olarak normal (gerçekçi
fundus) görüntü üretmeye devam eder — `||ΔG||` neredeyse HİÇ değişmez
(tek bir 320-boyutlu satır çifti takas edildi, milyonlarca parametreden
sadece ikisi) — ama DR sınıfı `row_a` istendiğinde `row_b`'nin (ve
tersi) görsel karakteristiğini üretir.

**Gerçek Colab bulgusu (H1 testi, `poisoned_alone` koşulu,
`docs/phase_f_attacks.md`'de tam analiz):** basit `swap_embed_rows`
zehirli site TEK BAŞINA ölçüldüğünde takas AÇIKÇA görünüyor (DR-4
istenince gerçek DR-0'a KID=0.0127, gerçek DR-4'e 0.0710 — 5 kat
fark) — ama 4 siteli FedAvg'dan SONRA bu sinyal SEYRELİYOR (aynı
hücre 0.0667'ye çıkıp kayboluyor), çünkü zehirli site katkının SADECE
1/4'ünü oluşturuyor, diğer 3 dürüst site ORİJİNAL satırı getiriyor.
`compensated_swap_embed_rows` bunu TELAFİ EDER — `scaled_poison.py`'nin
ΔG'yi ölçekleyerek AYNI seyrelmeyi telafi etme mantığının embed
satırına uygulanmış hali.

Bu yüzden İKİ varyant da (`swap_embed_rows`/basit,
`compensated_swap_embed_rows`/telafili) TUTULUYOR — hangisinin norm
kontrolünü aşıp aşmadığı (basit: muhtemelen AŞMAZ; telafili: 4 kat
daha büyük bir ΔG üretir, AŞABİLİR — bkz. `docs/phase_f_attacks.md`)
makalenin "tespit edilebilirlik vs gizlilik" dengesini gösteren ayrı
bir karşılaştırma.

Asıl zarar (her iki varyant için de) `eval/metrics.py:
class_confusion_matrix`'in ölçtüğü SINIF TUTARLILIĞINDA ortaya çıkar,
FID muhtemelen DÜŞMEZ (görüntüler hâlâ gerçekçi, sadece YANLIŞ sınıfa
etiketli).

Saf state_dict operasyonudur — hiçbir model eğitilmez (CLAUDE.md
madde 2/3/6); ağırlık TAKASI/kompanzasyonu bir eğitim adımı DEĞİLDİR.
"""

from __future__ import annotations

DEFAULT_EMBED_KEY = "mapping.embed.weight"


def _validated_embed_rows(state_dict: dict, row_a: int, row_b: int, embed_key: str):
    """`swap_embed_rows`/`compensated_swap_embed_rows`'un ORTAK
    doğrulaması — embed tablosunun var/2-boyutlu/satır-aralığında
    olduğunu kontrol edip tabloyu döner (DRY)."""
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
    return embed


def swap_embed_rows(state_dict: dict, row_a: int, row_b: int, *, embed_key: str = DEFAULT_EMBED_KEY) -> dict:
    """`state_dict[embed_key]`'in `row_a`/`row_b` satırlarını takas eder,
    GERİ KALAN her şeyi (embed tablosunun diğer satırları DAHİL) AYNEN
    korur. `state_dict`'in kendisi (sığ kopya dışında) mutasyona
    UĞRATILMAZ — çağıranın orijinal dict'i hâlâ geçerli kalır."""
    embed = _validated_embed_rows(state_dict, row_a, row_b, embed_key)

    swapped = embed.clone()
    swapped[[row_a, row_b]] = swapped[[row_b, row_a]]

    result = dict(state_dict)
    result[embed_key] = swapped
    return result


def compensated_swap_embed_rows(state_dict: dict, row_a: int, row_b: int, num_sites: int, *, embed_key: str = DEFAULT_EMBED_KEY) -> dict:
    """`swap_embed_rows`'un FedAvg-seyrelmesini TELAFİ EDEN hali.

    Zehirli site FedAvg'a `1/num_sites` ağırlıkla katkıda bulunuyor,
    diğer `num_sites-1` dürüst site DEĞİŞMEMİŞ (orijinal) satırı
    getiriyor. Basit takasta ortalama sonuç hedefe SADECE
    `1/num_sites` oranında yaklaşır (gerçek Colab ölçümü:
    `docs/phase_f_attacks.md`'deki H1 testi). Bu fonksiyon
    `poisoned_row`'u, FedAvg SONRASI ortalama TAM OLARAK hedef satıra
    ulaşacak şekilde ÇÖZER:

        (num_sites-1)*honest_row_a + poisoned_row_a = num_sites*honest_row_b
        => poisoned_row_a = num_sites*honest_row_b - (num_sites-1)*honest_row_a

    (ve simetrik olarak `row_b` için). `honest_row_a`/`honest_row_b`:
    zehirli site'ın KENDİ (henüz değiştirilmemiş) satırları — diğer
    dürüst sitelerin AYNI round'da AYNI (ya da çok yakın) satırlara
    sahip olduğu varsayılıyor (hepsi aynı önceki global'den başlıyor);
    bu VARSAYIM `docs/phase_f_attacks.md`'de AÇIKÇA not düşülüyor.

    `num_sites < 2` anlamsız (seyrelme olmadan telafi gerekmez) —
    net hata verir."""
    embed = _validated_embed_rows(state_dict, row_a, row_b, embed_key)
    if num_sites < 2:
        raise ValueError(f"num_sites en az 2 olmalı (tek site FedAvg'da seyrelme yok), alınan: {num_sites}")

    honest_row_a = embed[row_a].clone()
    honest_row_b = embed[row_b].clone()

    compensated = embed.clone()
    compensated[row_a] = num_sites * honest_row_b - (num_sites - 1) * honest_row_a
    compensated[row_b] = num_sites * honest_row_a - (num_sites - 1) * honest_row_b

    result = dict(state_dict)
    result[embed_key] = compensated
    return result
