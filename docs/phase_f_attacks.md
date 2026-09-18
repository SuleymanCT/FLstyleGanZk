# Faz F — Saldırı × Koruma Matrisi (FID/KID/Sınıf-Tutarlılığı)

**Durum: kod yazıldı, Colab'da HENÜZ koşulmadı/doğrulanmadı.** Bu
doküman, `scripts/run_attacks.py`'nin üreteceği
`{zk_root}/attacks/attack_results.json`'dan doldurulacak bir
ŞABLONDUR — CLAUDE.md madde 6 gereği hiçbir FID/KID/sınıf-tutarlılığı
sayısı burada UYDURULMAZ, gerçek Colab koşumundan SONRA doldurulur.

## Amaç

ZK doğrulamasının + norm kontrolünün zehirleme saldırılarını
GERÇEKTEN engelleyip engellemediğini, ölçülebilir FID/KID/sınıf-
tutarlılığı metrikleriyle göstermek. Üç saldırı, DÖRT sitenin biri
üzerinde, AĞIRLIK SEVİYESİNDE simüle edilir — hiçbir model eğitilmez
(CLAUDE.md madde 2/3/6): `attacks/random_weights.py`,
`attacks/scaled_poison.py` (10x/50x/100x), `attacks/conditional_poison.py`.

## Tasarım kararları (kullanıcı onayına sunulan sapmalar)

### 1. Sınıf-tutarlılığı metriği: 2-sınıflı kontrolden 5×5 karışıklık matrisine genişletildi

Kullanıcının önerisi (DR-0 üretimini gerçek DR-4 ile KID üzerinden
karşılaştırmak) doğru yöndeydi ama SADECE ÖNCEDEN BİLİNEN saldırı
çiftine (0↔4) özgüydü. Bunun yerine `eval/metrics.py: class_confusion_matrix`
TÜM DR sınıfları (0-4) için `kid_matrix[g][r]` = "sınıf g için ÜRETİLEN
görüntüler" vs "sınıf r'ye ait GERÇEK görüntüler" KID'i hesaplıyor —
5×5'lik tam bir matris. `perceived_class[g] = argmin_r kid_matrix[g][r]`
ile üretilen her sınıfın GERÇEKTE hangi sınıfa en yakın olduğu
bulunuyor; `perceived_class[g] != g` bir SINIF KARIŞIKLIĞI/TAKASI
TESPİTİ. Bu, HANGİ iki sınıfın takas edildiğini ÖNCEDEN bilmeye
gerek kalmadan HERHANGİ bir swap deseni yakalar, ve matrisin köşegeni
zaten kullanıcının istediği "sınıf başına KID"yi veriyor. Aynı
Inception özellik uzayı (FID/KID'in kullandığı, `DETECTOR_URL`) ve
StyleGAN-XL'in KENDİ `kernel_inception_distance.compute_kid` formülü
(WebFetch ile doğrulanmış, `eval/metrics.py: compute_kid_from_features`
olarak BİREBİR aynı) kullanılıyor — yeni bir metrik İCAT EDİLMEDİ,
mevcut FID/KID makinesinin sınıf-filtrelenmiş bir uygulaması.

### 2. ZK "yakaladı mı" sorusu — kod incelemesiyle DÜZELTİLEN bir varsayım

Kullanıcının orijinal beklentisi: "conditional_poison ZK'nın asıl
değerini gösterir" (norm'un kaçırdığı bir şeyi ZK'nın yakalaması
beklentisiyle). **`orchestrator/round_runner.py: generate_and_submit_proof`/`run_round`
incelendiğinde bu beklenti DOĞRULANAMADI:** weight commitment
(`compute_weight_commitment`) ve ZK devresi HER ZAMAN AYNI
`full_state` objesinden türetiliyor — bu kod tabanında bir sitenin
"taahhüt ettiğinden FARKLI bir ağırlığı ispatlayıp katkıda bulunması"
için HİÇBİR kod yolu yok (tek, dürüst istemci akışı var). Yani ZK
GERÇEKTEN, ağırlığın İÇERİĞİNİ (iyi/kötü) DEĞİL, taahhüt→ispat→katkı
ÜÇLÜSÜNÜN TUTARLILIĞINI kanıtlıyor. Üç saldırı da (dürüst ama
BOZULMUŞ bir istemci modelinde) bu tutarlılığı BOZMUYOR — bu yüzden
ZK'nın hiçbirini yakalaması BEKLENMEZ (`attacks/detection.py:
verify_commitment_consistency` bunu VARSAYMAZ, her saldırı için
GERÇEKTEN hesaplar — üçünde de sonuç aynı çıkarsa bu KANITLANMIŞ
olur, uydurulmamış olur). Ayrıca ZK devresi SADECE mapping alt-ağını
kapsıyor (`param_visibility="fixed"`) — `random_weights`/`scaled_poison`
(TÜM `G`) synthesis ağını da bozuyor, ki bu HİÇBİR ZK devresinde YER
ALMIYOR (`attacks/detection.py: attack_touches_zk_proven_scope` bu
kapsam-içi/dışı ayrımını raporluyor). **Bu, kullanıcının orijinal
"ZK'nın asıl değeri conditional_poison'da görünür" hipotezini
İNCELTEN/DÜZELTEN bir bulgu** — gerçek sonuç muhtemelen "ZK
HİÇBİRİNİ yakalamaz" olacak (Colab koşumuyla doğrulanacak).

### 3. Kapsam daraltması — "3 round canlı entegre koşu" bu turda YAPILMADI

PLAN.md'nin Faz F'nin eski/kısa taslağı bir "canlı entegre koşu"
maddesi içeriyordu; kullanıcının bu turdaki detaylı talimatı SADECE
saldırı×koruma matrisini kapsıyor, canlı koşudan hiç bahsetmiyor. Bu
yüzden o madde AYRI bir iş olarak PLAN.md'de bırakıldı (bu fazın
kapsamına DAHİL EDİLMEDİ) — istenirse ayrı bir istek olarak ele
alınabilir.

## Yöntem

1. `scripts/run_attacks.py`, `--round`'un 4 site'ının GERÇEK
   `G_ema.state_dict()`'ini (`fl.round_replay.load_site_update`) ve
   `--round - 1`'in GERÇEK `fedavg_{round-1}.pt`'sini
   (`scripts.audit_fedavg.load_fedavg_file`) yükler — hiçbiri
   uydurulmaz/simüle edilmez, Faz A'nın GERÇEK checkpoint'leri.
2. Bir site (`--poisoned-site`, varsayılan 0) seçilen saldırıyla
   bozulur.
3. **UNPROTECTED**: `fl.fedavg_utils.RunningAverage` ile kapısız
   FedAvg (4 site, zehirli dahil).
4. **PROTECTED**: `orchestrator.aggregate.aggregate_round` (Faz D'de
   ZATEN yazılmış/test edilmiş) — ZK-onay + norm kapısı, zehirli site
   norm eşiğini aşarsa dışlanır.
5. Her iki koşulun sonuç `G_ema`'sı üzerinde `eval/metrics.py` ile
   `fid50k_full`/`kid50k_full` (StyleGAN-XL'in KENDİ `metric_main.calc_metric`'i,
   `training_options.json`'dan okunan GERÇEK ayarlarla) + 5×5 sınıf-
   tutarlılığı matrisi hesaplanır.

## ANA TABLO (Colab koşumundan sonra doldurulacak)

| saldırı | koşul | ||ΔG|| | tau aşıldı mı (norm) | ZK kapsamda mı | ZK yakaladı mı | FID (fid50k_full) | KID(ort, köşegen) | sınıf tutarlılığı (swap tespit edilen sınıf sayısı) | zehirli site dahil mi |
|---|---|---|---|---|---|---|---|---|---|
| random_weights | unprotected | — | — | — | — | — | — | — | evet (kapısız) |
| random_weights | protected | — | — | — | — | — | — | — | — |
| scaled_poison_10x | unprotected | — | — | — | — | — | — | — | evet (kapısız) |
| scaled_poison_10x | protected | — | — | — | — | — | — | — | — |
| scaled_poison_50x | unprotected | — | — | — | — | — | — | — | evet (kapısız) |
| scaled_poison_50x | protected | — | — | — | — | — | — | — | — |
| scaled_poison_100x | unprotected | — | — | — | — | — | — | — | evet (kapısız) |
| scaled_poison_100x | protected | — | — | — | — | — | — | — | — |
| conditional_poison | unprotected | — | — | — | — | — | — | — | evet (kapısız) |
| conditional_poison | protected | — | — | — | — | — | — | — | — |

(Referans: mevcut deneyin korumasız FID'i `13.13` — `fid50k_full`,
BİREBİR aynı ölçüm ayarlarıyla karşılaştırılabilir olması gerekiyor.)

## Beklenen (ama VARSAYILMAYAN, Colab'da doğrulanacak) desen

- `random_weights`/`scaled_poison_50x`/`scaled_poison_100x`: `||ΔG||`
  büyük olasılıkla `tau=3000`'i AŞAR — norm kontrolü YAKALAR, zehirli
  site protected koşulda dışlanır.
- `scaled_poison_10x`: sınırda olabilir — GERÇEK `||ΔG||` ölçülecek,
  varsayılmıyor.
- `conditional_poison`: `||ΔG||` NEREDEYSE HİÇ değişmez (tek bir
  320-boyutlu satır çifti) — norm kontrolü büyük olasılıkla KAÇIRIR.
  FID muhtemelen düşmez (görüntüler hâlâ gerçekçi). Sınıf-tutarlılığı
  matrisinde `perceived_class[0]==4` ve `perceived_class[4]==0`
  (swap) BEKLENİYOR — bu GERÇEKLEŞİRSE, saldırı hem norm hem
  (madde 2'deki analiz gereği) ZK tarafından KAÇIRILMIŞ ama
  istatistiksel/davranışsal analizle YAKALANMIŞ olacak.
- **Madde 2'nin analizi doğruysa**: `zk_caught` SÜTUNU beş saldırının
  BEŞİ İÇİN DE `False` olacak — bu, "ZK içerik değil tutarlılık
  kanıtlar" bulgusunun doğrudan kanıtı, ZK'nın olduğundan güçlü
  gösterilmediğinin garantisi.

## Dürüstlük noktası (makalede aynen yer almalı)

ZK doğrulamasının bu sistemdeki GERÇEK rolü — taahhüt-ispat-katkı
tutarlılığı, İÇERİK/kalite denetimi DEĞİL — GERİLİMLİ ama ÖNEMLİ bir
sınırlılık: zincir üzeri ZK katmanı TEK BAŞINA zehirlemeye karşı
yeterli DEĞİL, istatistiksel/davranışsal analiz (FID/KID/sınıf-
tutarlılığı gibi) TAMAMLAYICI bir savunma katmanı olarak GEREKLİ. Bu
ayrım makalenin "ZK'nın sınırlılıkları" bölümüne AÇIKÇA girmeli —
ZK'yı olduğundan güçlü göstermemek adına.
