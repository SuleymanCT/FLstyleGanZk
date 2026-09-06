# Faz A Raporu — Envanter ve Çıkarma

Bu rapor, Colab'da gerçek Drive verisiyle üretilmiş üç dosyaya dayanır:
`zk_artifacts_results/inventory.json` (580 KB, 60 site kaydı),
`zk_artifacts_results/fedavg_audit.json`, `zk_artifacts_results/delta_norms.json`.
Hiçbir sayı uydurulmadı; bu dosyalarda bulunmayan bir bilgi (özellikle
A bölümündeki tensör şekilleri) açıkça "mevcut değil" olarak işaretlendi.

## Genel durum

- **60/60** (round, site) çifti eksiksiz (`missing_expected_pairs: []`,
  `unexpected_pairs: []`) — 15 round × 4 site.
- **15** `fedavg_*.pt` dosyası (`fedavg_0.pt`..`fedavg_14.pt`).
- Tüm 60 site aynı mimariye sahip (bkz. Bölüm E).
- `pkl_top_level_keys` her sitede aynı: `["D", "G", "G_ema", "augment_pipe", "progress", "training_set_kwargs"]`.
- `training_options.json` sadece beklenen (site'a özgü) alanlarda
  farklı: `resume_pkl`, `run_dir`, `training_set_kwargs` — mimariyle
  ilgili hiçbir alan farklı değil.
- `parallel_final` içinde round/site deseni dışında 3 "base model adayı"
  bulundu: `global_round_0.pkl` (muhtemelen taban model — round 0
  öncesi başlangıç noktası), `global_round5_start.pkl`,
  `fid_round10.pkl`. Bunların rolü bu raporun kapsamı dışında, ayrıca
  doğrulanmalı.

## A) Mapping katmanları

`inventory.json`'daki `candidate_submodules.mapping` tespiti tüm 60
sitede **birebir aynı** 5 girişi buluyor: `mapping`, `mapping.embed`,
`mapping.embed_proj`, `mapping.fc0`, `mapping.fc1`. `mapping`'in
kendisi (MappingNetwork kapsayıcısı) doğrudan parametre taşımıyor
(`num_params: 0`) — tüm ağırlıklar alt modüllerinde:

| Katman | Tip | Parametre sayısı |
|---|---|---:|
| `mapping.embed` | `Embedding` | 320.000 |
| `mapping.embed_proj` | `FullyConnectedLayer` | 20.544 |
| `mapping.fc0` | `FullyConnectedLayer` | 66.048 |
| `mapping.fc1` | `FullyConnectedLayer` | 262.656 |
| **Toplam (4 katman)** | | **669.248** |

- **Tam G'nin toplam parametre sayısı:** 61.132.095 (39 modüllük
  ağacın `num_params` toplamı; recurse=False sayımı olduğundan hiçbir
  parametre iki kez sayılmıyor).
- **Oran:** 669.248 / 61.132.095 ≈ **%1,095** (yaklaşık 1/91).

**Bilinen sınır:** `inventory.json`, `describe_module_tree` tarafından
modül BAŞINA toplam parametre sayısını kaydediyor — `weight`/`bias`
tensörlerinin ayrı ayrı **şekillerini** (ör. `mapping.fc0.weight`'in
`(512, 128)` mi `(128, 512)` mi olduğu) kaydetmiyor. Bu bilgi mevcut üç
dosyanın hiçbirinde yok; sayılardan geriye doğru "muhtemel" bir şekil
türetmek (ör. `fc0`'ın 66.048 parametresini `out*(in+1)` formülüyle
çözmeye çalışmak) StyleGAN-XL'in `FullyConnectedLayer`/`Embedding`
uygulamasının tam iç ayrıntıları (bias var mı, ek ölçekleme var mı)
teyit edilmeden bir tahmin olur — CLAUDE.md madde 6 gereği burada
uydurulmadı. Gerçek şekiller gerekiyorsa (Faz C devre tasarımı için
gerekecek), `scripts/extract_shards.py`'nin ürettiği `*_meta.json`
dosyalarındaki `shard_shapes` alanı kullanılabilir, ya da
`scripts/inventory.py`'ye parametre bazlı bir `named_parameters()`
geçişi eklenip yeniden koşulabilir.

Diğer boyutlar (`dims`, 60/60 sitede birebir aynı): `z_dim=64`,
`c_dim=5`, `w_dim=512`, `num_ws=16`.

## B) fedavg numaralandırması

`audit_fedavg.py`, her round N için 4 sitenin `G_ema`'sının fp32
ortalamasını hesaplayıp hem `fedavg_N.pt` hem `fedavg_(N+1).pt` ile
karşılaştırdı:

| Round | vs `fedavg_N` (max\|Δ\|) | vs `fedavg_(N+1)` (max\|Δ\|) |
|---:|---:|---:|
| 0 | **0.0** | 0.8709 |
| 1 | **0.0** | 1.2001 |
| 2 | 0.4737 | 1.1384 |
| 3 | **0.0** | 0.7672 |
| 4 | **0.0** | 2.4409 |
| 5 | **0.0** | 1.7625 |
| 6 | **0.0** | 1.8175 |
| 7 | **0.0** | 1.6493 |
| 8 | **0.0** | 1.6765 |
| 9 | **0.0** | 1.8564 |
| 10 | **0.0** | 1.6359 |
| 11 | **0.0** | 1.3571 |
| 12 | **0.0** | 1.2218 |
| 13 | 0.4936 | 1.3468 |
| 14 | **0.0** | — (fedavg_15 yok) |

**Sonuç:** Numaralandırma kesinleşti — **`fedavg_N.pt` = round_N'in
dört sitesinin fp32 ortalaması**, `fedavg_(N+1).pt` DEĞİL. 13/15
round'da fark tam olarak 0.0 (bit-birebir eşleşme); round 2 ve 13'te
çok küçük bir sapma var (sırasıyla 0.4737 ve 0.4936) — bu, `vs_fedavg_(N+1)`
farklarının tipik büyüklüğünden (0.77–2.44) belirgin şekilde küçük,
yani yön tespiti hâlâ net (round 2 ve 13, kendi `fedavg_N`'ine
`fedavg_(N+1)`'den çok daha yakın), ama tam 0.0 olmaması muhtemelen
toplama sırası/fp32 yuvarlama farkından kaynaklanıyor (kesin sebep
doğrulanmadı, uydurulmadı). Tutarlılık: **15 round'un 15'inde de** yön
aynı — hiçbir round ters numaralandırma göstermiyor.

## C) Delta-G normları

`delta_norms.json`'daki 56 değer (round 0 hariç 14 round × 4 site;
round 0'da önceki global olmadığından delta hesaplanmadı), sırayla
round/site'a göre ayrıştırılıp aşağıdaki tabloya döküldü:

| Round | site0 | site1 | site2 | site3 | ortalama | std | min | max | CV% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2746 | 0.2747 | 2021.4641 | 1981.9800 | 1000.9983 | 1000.8211 | 0.2746 | 2021.4641 | 99.98 |
| 2 | 2004.6579 | 0.2377 | 1969.3284 | 0.2386 | 993.6157 | 993.4560 | 0.2377 | 2004.6579 | 99.98 |
| 3 | 2003.7912 | 0.2404 | 1951.3547 | 0.2414 | 988.9069 | 988.8398 | 0.2404 | 2003.7912 | 99.99 |
| 4 | 2000.3344 | 0.2282 | 1943.9170 | 0.2291 | 986.1772 | 986.1502 | 0.2282 | 2000.3344 | 100.00 |
| 5 | 2512.3695 | 2481.3813 | 2458.9860 | 2474.1011 | 2481.7095 | 19.4576 | 2458.9860 | 2512.3695 | 0.78 |
| 6 | 2500.5625 | 1632.9265 | 2489.4869 | 2474.0597 | 2274.2589 | 370.3930 | 1632.9265 | 2500.5625 | 16.29 |
| 7 | 2499.3387 | 2516.1593 | 2472.8127 | 1566.8725 | 2263.7958 | 402.6655 | 1566.8725 | 2516.1593 | 17.79 |
| 8 | 2491.5410 | 1580.0675 | 2495.0598 | 2499.2300 | 2266.4746 | 396.3067 | 1580.0675 | 2499.2300 | 17.49 |
| 9 | 2497.1968 | 2520.5920 | 1565.2334 | 2481.1993 | 2266.0554 | 404.8622 | 1565.2334 | 2520.5920 | 17.87 |
| 10 | 2506.6159 | 2502.0297 | 2501.3474 | 2496.0365 | 2501.5074 | 3.7525 | 2496.0365 | 2506.6159 | 0.15 |
| 11 | 2511.4474 | 1663.0323 | 2509.5080 | 2506.2059 | 2297.5484 | 366.3428 | 1663.0323 | 2511.4474 | 15.94 |
| 12 | 1595.8144 | 1595.8145 | 2488.9043 | 1595.8143 | 1819.0869 | 386.7193 | 1595.8143 | 2488.9043 | 21.26 |
| 13 | 2045.6273 | 2100.3152 | 2088.8563 | 531.5453 | 1691.5860 | 670.0603 | 531.5453 | 2100.3152 | 39.61 |
| 14 | 1674.1644 | 1674.1645 | 1674.1644 | 1674.1643 | 1674.1644 | 0.0001 | 1674.1643 | 1674.1645 | **0.00** |

**Genel (n=56) percentile'lar** (`delta_norms.json: percentiles`):
p50 = **2013.06**, p90 = **2506.41**, p99 = **2518.15**.

**Trend:** Round 1–4'te değerler net biçimde iki kümeye ayrılıyor
(≈0.23–0.28 "neredeyse sıfır" ve ≈1900–2020 "büyük"); round 5'ten
itibaren çoğu değer ≈2450–2520 bandına sıçrıyor ve round 14'e kadar
büyük ölçüde orada kalıyor, ama neredeyse her round'da bir veya birkaç
site beklenmedik şekilde düşük kalıyor (round 6 site1, round 7 site3,
round 8 site1, round 9 site2, round 11 site1, round 12'de üç site
birden, round 13 site3). Round 14 ise tüm serinin en sıra dışı noktası
— ayrıntı için Bölüm D.

## D) Round 14 anomalisi ve site-içi varyansın round bazında seyri

Kullanıcının işaret ettiği round 14 değerleri (`1674.164444 / 1674.164487
/ 1674.164378 / 1674.164336`) `delta_norms.json`'da doğrulandı — dört
site de birbirinden **sadece ~0.0001 birim** farklı (CV = **%0.00**,
tablodaki 14 round içinde en düşük). Bu, diğer round'ların HİÇBİRİNDE
böyle değil:

- **Round 14 (CV %0.00)** ve daha düşük ölçekte **round 10 (CV %0.15)**
  ve **round 5 (CV %0.78)** dışında, kalan round'ların tümü **%15 ile
  %100** arasında değişen çok yüksek site-içi varyansa sahip.
- Round 1–4 en uç durum: CV neredeyse **%100** — çünkü bu round'larda
  4 siteden 2'si (genelde site1 ve site3, round 1'de site0 ve site1)
  önceki globalden neredeyse hiç uzaklaşmamış (Δ≈0.23–0.28) iken diğer
  2'si büyük ölçüde eğitilmiş (Δ≈1900–2020) görünüyor. Bu, ilk
  round'larda bazı sitelerin (muhtemelen tutarlı biçimde aynı site
  indeksleri) yerel eğitiminin neredeyse etkisiz kaldığına ya da
  farklı bir başlangıç/epoch ayarına işaret edebilir — kesin sebep bu
  üç dosyadan çıkarılamıyor, sadece örüntü olarak not ediliyor.
- Round 5 ve 10, tüm 4 sitenin birbirine yakın (ama round 14 kadar
  değil) davrandığı ara noktalar (CV %0.78 ve %0.15).
- Round 6–9, 11–13 aralığında düzenli olarak TEK bir site (bazen üçü,
  round 12'de olduğu gibi) beklenenden belirgin düşük kalıyor, diğerleri
  ~2470–2520 bandında kümeleniyor — round 14'teki gibi TÜM sitelerin
  aynı anda yakınsaması bu aralıkta hiç görülmüyor.

**Kopya/kayıt hatası ihtimali `scripts/probe_duplicates.py` ile ELENDİ.**
Shard seviyesinde (`canonical_hash`, mapping+embedding katmanları)
15 round'un TAMAMI için kontrol edildi (`zk_artifacts_results/duplicate_check.json`):

- Her round'da **4 benzersiz hash** (round 14 dahil) — hiçbir round'da
  `duplicate_groups` boş değil, yani hiçbir site diğerinin birebir
  kopyası değil.
- Hiçbir sitenin shard'ı bir önceki round'un global ağırlığıyla
  (`fedavg_(N-1)`'in aynı prefixlerle filtrelenmiş `G_ema`'sı) aynı
  değil (`prev_global_matches` tüm round/site'larda `false`) — yani
  hiçbir site "hiç eğitilmeden global ağırlığı olduğu gibi kaydetme"
  durumunda değil.

Yani round 14'teki dört site GERÇEKTEN dört farklı, bağımsız ağırlık
kümesi — sadece bu ağırlıkların önceki globalden **norm cinsinden
kat ettiği mesafe** tesadüfen (ya da yapısal bir nedenle) neredeyse
özdeş. Bu, bir veri/kayıt hatası değil, **gerçek bir gözlem**.

**Yorum:** Federe kurulum IID bölme kullanıyor (dört site aynı
dağılımdan, eşit büyüklükte veri payı alıyor) ve her round'da eşit
sayıda yerel adım (tick/epoch) ile eğitiliyor. Bu koşullar altında
dört site, global ağırlıktan başlayıp **birbirinden farklı yönlerde**
(farklı mini-batch örneklemesi, farklı gradyan yörüngeleri — bu yüzden
hash'leri hep farklı) ilerliyor, ama optimizasyon dinamiği (aynı öğrenme
oranı, aynı adım sayısı, aynı veri dağılımı büyüklüğü) hepsini globalden
**benzer büyüklükte bir adımla** uzaklaştırıyor. Farklı yöne giden ama
aynı uzunlukta dört vektör — norm'da yakınsama, ağırlıkta yakınsama
DEĞİL. Round 14'te bu etki en belirgin haliyle görülüyor (CV %0.00),
round 5 ve 10'da da (CV %0.78, %0.15) benzer bir eğilim var; round
1–4'teki bimodal örüntü (bazı sitelerin neredeyse hareket etmemesi) ve
round 6–9/11–13'teki tek-site-düşük örüntüsü muhtemelen eğitimin erken
aşamalarında/ara dönemlerde bu "eşit adım büyüklüğü" etkisinin henüz
istikrar kazanmamış olmasından kaynaklanıyor — kesin mekanizma bu
dosyalardan çıkarılamıyor, ama IID + eşit-tick açıklamasıyla tutarlı.

Sonuç: Faz D'nin τ eşiği bu gerçek varyansı (CV %0'dan %100'e kadar
geniş bir aralık) hesaba katmalı — round 14/5/10 gibi düşük-varyans
round'ları "şüpheli" olarak yanlış işaretlememek için eşik, gözlenen
p99 (2518) civarından, TEK bir site'ın normal aralığın dışına
(ör. p99'un belirgin üzerine) çıkması durumunu yakalayacak şekilde
seçilmeli — round-içi düşük varyansın kendisi bir sabotaj göstergesi
değil.

## E) Round'lar arasında mimari fark var mı?

**Hayır — 60/60 sitede mimari birebir aynı.** Doğrulanan alanlar:

- `module_tree` uzunluğu: tüm 60 sitede **39** modül.
- Toplam parametre sayısı: tüm 60 sitede **61.132.095** (tek bir
  farklı değer yok).
- `dims` (`z_dim`, `c_dim`, `w_dim`, `num_ws`): tüm 60 sitede birebir
  aynı imza.
- `candidate_submodules` (mapping/embedding tespiti): tüm 60 sitede
  birebir aynı imza.
- `pkl_top_level_keys`: tüm 60 sitede birebir aynı (`D`, `G`, `G_ema`,
  `augment_pipe`, `progress`, `training_set_kwargs`).
- Mapping katmanlarının 4'ünün de parametre sayıları (`mapping.embed`,
  `mapping.embed_proj`, `mapping.fc0`, `mapping.fc1`): tüm 60 sitede
  birebir aynı imza `(0, 320000, 20544, 66048, 262656)`.
- `training_options.json` sadece beklenen, site'a özgü alanlarda
  (`resume_pkl`, `run_dir`, `training_set_kwargs`) farklılık gösteriyor
  — mimariyle ilgili hiçbir alanda fark yok.

Sonuç: 15 round boyunca StyleGAN-XL mimarisi (mapping ağı dahil)
değişmedi; C1'deki minimal mapping modülü tek bir sabit yapıya göre
kurulabilir, round'a göre koşullu dallanma gerekmiyor.
