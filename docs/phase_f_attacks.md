# Faz F — Saldırı × Koruma Matrisi (norm kapısı / ZK / sınıf-tutarlılığı)

**Durum: quick mod Colab'da TAMAMLANDI (6 saldırı × 3 koşul). Norm/ZK
sonuçları ve sınıf-tutarlılığı matrisleri GERÇEK Colab ölçümüdür. FID/KID
ÖLÇÜLMEDİ** (`ref` modunda `fid50k_full` koşul başına ~13,8 saat —
bkz. "Sınırlılıklar"). Hiçbir sayı uydurulmadı (CLAUDE.md madde 6); bu
dokümandaki tüm rakamlar konsol/JSON çıktısından alınmıştır.

## ANA SONUÇ (özet)

**Koşullu (sınıf-hedefli) zehirleme, FedAvg seyrelmesini telafi edecek
kadar güçlendirilse BİLE norm tabanlı savunmanın görüş alanı dışında
kalıyor.** Telafili saldırının ‖ΔG‖'si 2519,53 (tau=3000'in ALTINDA); saldırının
kendi katkısı yalnızca 255,33, sitenin doğal eğitim kaymasının (2506,64)
yanında gürültü seviyesinde — toplam normu %0,5 büyütüyor. `tau`
keyfi değil, Faz A'nın gerçek ‖ΔG‖ dağılımından (p99=2518, n=56)
seçildi: eşiği bu saldırıyı yakalayacak kadar (yaklaşık 2519'un altına)
düşürmek meşru güncellemelerin ~%1'ini (p99'un tanımı gereği) — ve bu
sitenin kendi doğal kaymasını (2506,64) — de eleyecektir. Savunmanın ZK
kısmı ise bu saldırıyı yakalamıyor ve mimari olarak yakalaması da
BEKLENMİYOR (aşağıda).

## Amaç

ZK doğrulamasının + norm kontrolünün zehirleme saldırılarını
GERÇEKTEN engelleyip engellemediğini, ölçülebilir FID/KID/sınıf-
tutarlılığı metrikleriyle göstermek. Altı saldırı varyantı, DÖRT sitenin biri
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

### 4. CUDA custom op derlemesi başarısız — `ref` moduna geçiş

Colab koşumunda StyleGAN-XL'in `bias_act`/`upfirdn2d`/`filtered_lrelu`
CUDA-derlemeli custom op'ları derlenemedi (`ModuleNotFoundError: No
module named 'bias_act_plugin'`, ninja+`CUDA_HOME` mevcut olmasına
RAĞMEN — Colab'ın Python 3.13/güncel PyTorch kombinasyonunun bilinen
bir sorunu). WebFetch ile gerçek kaynak (`torch_utils/ops/bias_act.py`/
`upfirdn2d.py`/`filtered_lrelu.py`) doğrulandı: `_init()`'i yamalamak
kırılgan olurdu (bu depoda `_init()` derleme başarısız olsa bile
`False` DÖNMÜYOR — hata `bias_act()` çağrısına kadar yükseliyor,
tam gözlenen hata BUDUR). Bunun yerine `eval.metrics.force_stylegan_ops_impl`
üç modülün genel işlevini SARMALAYIP `impl` argümanını HER ÇAĞRIDA
zorluyor (internal katman kodu `impl=` kwarg'ını hiç geçmiyor, hep
`'cuda'` varsayılanına güveniyor). `--ops-impl auto` (varsayılan)
küçük bir deneme üretimiyle CUDA'yı test edip BAŞARISIZ olursa
UYARIYLA `ref`'e düşüyor; `ops_impl` alanı sonuç JSON'una kaydediliyor.

### 5. `ref` modunda CUDA belleği tükendi — küçük batch + OOM-yeniden-deneme + CPU fallback

`ref` moduna geçince BAŞKA bir sorun ortaya çıktı: `upfirdn2d._upfirdn2d_ref`
içindeki `F.pad`, StyleGAN3-r'ın geniş filtreleriyle TEK bir görüntü
için bile devasa bir ara tensör istedi (`13.37 GiB`, 39.49 GiB'lık
GPU'da). Bu, CUDA'nın fused kernellerinin TEK bir işlemde yaptığını
`ref` yolunun birden fazla büyük ara tensörle (unfused) yapmasının
DOĞRUDAN sonucu — StyleGAN3'ün `ref` yolunun bilinen bir bellek
maliyeti. Düzeltmeler:

- **`--gen-batch-size`** (varsayılan: `ops_impl`'e göre otomatik —
  `ref`→1, `cuda`→32) — görüntü üretim batch boyutunu kontrol eder.
- **`eval.metrics.generate_class_images`** artık OOM'u (`is_cuda_oom_error`
  — hem yeni `torch.cuda.OutOfMemoryError` hem eski `RuntimeError`
  mesaj-tabanlı tespiti, PyTorch sürümü VARSAYILMIYOR) yakalayıp
  `torch.cuda.empty_cache()`+`gc.collect()` sonrası batch boyutunu
  YARIYA indirip AYNI chunk'ı yeniden dener (en fazla 4 kez); batch
  1'de bile OOM olursa net bir hata + `--device cpu` önerisiyle durur
  (sessizce yutulmaz). Üretim zaten `torch.no_grad()` içindeydi
  (gradyan grafiği hiç tutulmuyor); her batch sonrası ara tensörler
  `del` edilip GPU önbelleği boşaltılıyor.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (hata
  mesajının önerdiği, bellek parçalanmasını azaltan ayar) `torch`
  import edilmeden ÖNCE, script'in en başında ayarlanıyor.
- **`--device {auto,cuda,cpu}`** — batch=1'de bile OOM olursa CPU
  fallback'i (çok daha yavaş, ölçülüp `estimate_full_mode_cost` ile
  raporlanıyor — `device=cpu`ysa StyleGAN-XL'in `impl` dallanması
  zaten CPU tensöründe HER ZAMAN `ref` yoluna düştüğünden `resolve_ops_impl`
  smoke-testi atlayıp doğrudan `ops_impl='ref'` döner).
- **Değerlendirilen ama GEREKSİZ bulunan önlem:** "4 sitenin state_dict'i
  + G_ema shell GPU'da mı duruyor" sorusu kod incelemesiyle netleştirildi
  — `fl.stylegan_xl_env.load_network_pkl`/`legacy.load_network_pkl` hiçbir
  `.cuda()` çağrısı YAPMIYOR, yüklenen state_dict'ler ve `g_ema_shell`
  varsayılan olarak CPU'da kalıyor; `evaluate_condition` SADECE aktif
  kullanılan `g_ema_shell`'i `.to(device)` ile GPU'ya taşıyor (TEK kopya,
  koşullar arasında YENİDEN tahsis edilmiyor). Yani OOM'un kaynağı
  BİRİKEN site ağırlık kopyaları DEĞİL, sadece `ref` yolunun TEK bir
  ileri geçişteki ara tensör boyutuydu — bu yüzden "yükleme sonrası
  CPU'ya taşı" için EK bir kod değişikliği GEREKMEDİ (zaten öyleydi).

### 6. `--ops-impl auto` smoke-test'i bir CİHAZ hatasıyla çöktü — CUDA aslında test edilmemişti

Gerçek hata (`RuntimeError: Expected all tensors to be on the same
device, but got index is on cuda:0, different from other tensors on
cpu (wrapper_CUDA__index_select)`) bir CUDA DERLEME hatası DEĞİLDİ —
`resolve_ops_impl`'in smoke-test'i `z`/`c`'yi `device`'a taşırken
`g_ema`'nın KENDİSİNİ hiç taşımıyordu (`g_ema.to(device)` YOKTU) —
model CPU'da kalırken girdi GPU'ya gidince embedding/pozisyonel
tablo gibi bir `index_select` çağrısı çöküyordu. Eski kod bu hatayı
GENİŞ bir `except Exception` ile yakalayıp "CUDA derlemesi başarısız"
diye YANLIŞ teşhis edip SESSİZCE `ref`'e düşüyordu — yani CUDA
GERÇEKTEN çalışıyor olabilirdi ama HİÇ test EDİLEMEMİŞTİ. Düzeltme:
`resolve_ops_impl` artık smoke-test'ten ÖNCE `g_ema.to(device)`
çağırıp `eval.metrics.assert_module_on_device` ile TÜM parametre/
buffer'ların hedef cihazda olduğunu doğruluyor; `is_device_mismatch_error`
bu SINIF bir hatayı (mesajında "same device" geçen `RuntimeError`)
gerçek derleme hatalarından AYIRT edip AYRI, net bir hata olarak
yükseltiyor — bir daha SESSİZCE yanlış teşhis edilip gizlenmiyor.
**SONUÇ (Colab'da doğrulandı, KAPANDI):** düzeltmeden sonra smoke-test
GERÇEK derleme hatasıyla düştü — `Setting up PyTorch plugin
"bias_act_plugin"... Failed!` / `ModuleNotFoundError: No module named
'bias_act_plugin'` (cihaz sorunu DEĞİL). CUDA custom op'ları bu
ortamda derlenemiyor; `ref` modu KALICI, bkz. "CUDA meselesi" bölümü.

### 7. `ref`'te `full` modu pratik değil (~138 saat) — `--metrics-mode custom` eklendi

Ölçülen `0.498 s/görüntü` (ref, quick mod) ile `estimate_full_mode_cost`
`--metrics-mode full`'un (5 saldırı × 2 koşul × 2 metrik × 50.000
görüntü) TOPLAM ~138 saat süreceğini gösterdi — YAPILAMAZ. Eklenen
alternatif: `--metrics-mode custom --fid-num-gen <N>` (varsayılan
`N=5000`, kullanıcının önerisi) — `eval.metrics.run_custom_fid_kid`
StyleGAN-XL'in KENDİ temel fonksiyonlarını (`frechet_inception_distance.compute_fid`,
`kernel_inception_distance.compute_kid`) DOĞRUDAN çağırır, SADECE
resmi `metric_main.calc_metric` sarmalayıcısının hardcoded
`num_gen=50000`'ini atlar (FID/KID mantığı yeniden YAZILMIYOR).
**Bu sonuç `13.13` referansıyla KARŞILAŞTIRILAMAZ** — dönüş
değerindeki `fid_comparable_to_reference: False` alanı ve her konsol
çıktısındaki "13.13 İLE KARŞILAŞTIRILAMAZ" ibaresi bunu HER ÇIKTIDA
açıkça taşır (kullanıcının "her çıktıda net belirt" isteği).

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


## Ana tablo — norm kapısı (tau=3000, GERÇEK ölçüm)

| saldırı | ‖ΔG‖ | tau'ya oran | norm kapısı | sonuç |
|---|---|---|---|---|
| random_weights | 59 592 | 19,9× | YAKALADI | zehirli site dışlandı |
| scaled_poison_10x | 25 066 | 8,4× | YAKALADI | zehirli site dışlandı |
| scaled_poison_50x | 125 332 | 41,8× | YAKALADI | zehirli site dışlandı |
| scaled_poison_100x | 250 664 | 83,6× | YAKALADI | zehirli site dışlandı |
| conditional_poison (basit takas) | 2 507 | 0,84× | GEÇTİ | site dahil; FedAvg'da sinyal seyreliyor |
| conditional_poison_compensated | 2 519,53 | 0,84× | GEÇTİ | site dahil; FedAvg'da sinyal korunuyor |

ZK sütunu bilerek ayrı yazılmadı: `zk_caught` altı saldırının HEPSİ için
`False` ve bu bir Colab ölçümü değil KOD GARANTİSİ
(`verify_commitment_consistency` bir state'i kendisiyle karşılaştırır) —
bkz. "ZK'nın doğru çerçevesi".

### Telafinin ‖ΔG‖ üzerindeki etkisi (aritmetik doğrulaması)

| | doğal kayma (saldırısız site) | saldırının kendi katkısı | toplam ‖ΔG‖ |
|---|---|---|---|
| conditional_poison_compensated | 2 506,64 | 255,33 | 2 519,53 |

√(2506,64² + 255,33²) ≈ 2519,5 — katkı doğal kaymaya yaklaşık DİK
(kareler toplamı) bindiği için toplam ancak %0,5 artıyor. Basit
takasın katkısı ≈ 255,33/4 ≈ 64 (kompanzasyon embed delta'sını tam
`NUM_SITES=4` kat büyütür, birim testle doğrulanmıştır), toplam 2507.
Sonuç: telafi saldırıyı 4× güçlendirirken norm-görünürlüğünü neredeyse
hiç artırmıyor.

## Koşullu zehirleme varyantlarının sınıf-tutarlılığı karşılaştırması

Ölçüt: DR-4 istendiğinde ÜRETİLEN görüntüler ile gerçek DR-0 / gerçek
DR-4 arasındaki KID (satır DR-4; düşük = benzer). Takas başarılıysa
DR-0'a olan KID, DR-4'e olandan KÜÇÜK olmalı (oran = KID(DR-4)/KID(DR-0) > 1).

| durum | DR-4 → gerçek DR-0 | DR-4 → gerçek DR-4 | oran | perceived_class[4] |
|---|---|---|---|---|
| basit takas, zehirli site TEK BAŞINA | 0,0127 | 0,0710 | 5,59 | 0 |
| basit takas, FedAvg sonrası | 0,0667 | 0,0846 | 1,27 | 0 |
| **telafili**, zehirli site TEK BAŞINA | 0,0096 | 0,0591 | 6,16 | 0 |
| **telafili**, FedAvg sonrası | 0,0453 | 0,1075 | **2,37** | 0 |

Okuma: (i) tek başına iki varyant da takası açıkça gösteriyor (5,6× /
6,2×). (ii) Basit takasın sinyali FedAvg'da 1,27'ye çöküyor (seyrelme —
H1, doğrulandı). (iii) Telafi FedAvg sonrası oranı 1,27 → 2,37'ye çıkarıp
sinyalin bir kısmını GERİ KAZANDIRIYOR — tam telafi (oranın tek-başına
değerine dönmesi) olmadı: formül ortalama GİRDİ satırını hedefe eşitler,
ama üretilen görüntüler mapping/synthesis'in geri kalanından da geçtiği
için KID uzayına doğrusal yansımıyor. (iv) `perceived_class[4]=0` dört
durumda da.

### Tam matrisler (satır = istenen sınıf, sütun = gerçek sınıf DR-0..DR-4)

**Telafili, FedAvg sonrası (unprotected; protected aynı girdi ve aynı
tohumla hesaplandığından — zehirli site norm kapısını geçip dahil
edildiği için — aynıdır):**

| istenen | DR-0 | DR-1 | DR-2 | DR-3 | DR-4 |
|---|---|---|---|---|---|
| DR-0 | 0,0789 | 0,0389 | 0,0471 | 0,0415 | 0,0474 |
| DR-1 | 0,0913 | 0,0627 | 0,0804 | 0,0868 | 0,0837 |
| DR-2 | 0,0834 | 0,0483 | 0,0573 | 0,0597 | 0,0630 |
| DR-3 | 0,0772 | 0,0428 | 0,0445 | 0,0366 | 0,0401 |
| DR-4 | 0,0453 | 0,0859 | 0,0955 | 0,1120 | 0,1075 |

**Telafili, zehirli site TEK BAŞINA (`poisoned_alone`):**

| istenen | DR-0 | DR-1 | DR-2 | DR-3 | DR-4 |
|---|---|---|---|---|---|
| DR-0 | 0,1151 | 0,0999 | 0,0649 | 0,0719 | 0,0708 |
| DR-1 | 0,0500 | 0,0172 | 0,0298 | 0,0390 | 0,0342 |
| DR-2 | 0,0616 | 0,0164 | 0,0206 | 0,0305 | 0,0299 |
| DR-3 | 0,0563 | 0,0176 | 0,0188 | 0,0121 | 0,0143 |
| DR-4 | 0,0096 | 0,0471 | 0,0487 | 0,0642 | 0,0591 |

**Basit takas, zehirli site TEK BAŞINA (H1 testi):**
`perceived_class=[1,1,1,3,0]`; DR-4 satırı `[0,0127, 0,0525, 0,0559, 0,0773, 0,0710]`,
DR-0 satırı `[0,0539, 0,0119, 0,0168, 0,0195, 0,0180]`.

### Gözlem: takasın iki yönü asimetrik

DR-4→DR-0 yönü net. DR-0 yönü değil: telafili `poisoned_alone`'da DR-0
üretimi KENDİ gerçek sınıfına 0,1151 (satırın EN YÜKSEĞİ) — yani DR-0
üretimi BOZULMUŞ ama DR-4'e KAYMAMIŞ (DR-4'e 0,0708). Basit takasta da
DR-0 satırında argmin DR-1 (`perceived_class[0]=1`). Olası neden
(DOĞRULANMADI, hipotez): DR-0 satırı DR-4'ün gömmesinden ekstrapole
edilerek (`4·e4 − 3·e0`) veri manifoldunun dışına itiliyor ve üretim
kalitesini kaybediyor; ayrıca DR-0 gerçek görüntüleri KID uzayında diğer
sınıflardan uzak (DR-0 sütunu çoğu satırda yüksek). Kanıt olarak DR-4
satırı kullanılmalı; DR-0 yönü ölçüm sınırlılığı olarak raporlanır.

## ZK'nın doğru çerçevesi

ZK bu saldırıyı YAKALAMIYOR ve yakalaması BEKLENMİYOR. Mimari olarak
ZK katmanı bir taahhüt–ispat–katkı TUTARLILIK denetimidir, içerik
denetimi değildir: weight commitment ve ezkl devresi aynı `full_state`'ten
türer (`orchestrator/round_runner.py`), dolayısıyla dürüst bir istemcinin
KENDİ (zehirli) ağırlığını taahhüt edip ispatlaması hep geçerli çıkar.
ZK'nın değeri şudur: norm kapısı ve diğer kontroller bir ağırlık
kümesini denetlerken, denetlenen şeyin GERÇEKTEN katkıda bulunulan şey
olmasını garanti eder (bait-and-switch'i imkânsız kılar). Yani ZK
norm-tabanlı savunmayı TAMAMLAR, yerine geçmez; içerik zehirlemesine
karşı koruma sağlamaz. Kapsam ayrıntısı (`attack_touches_zk_proven_scope`):
rastgele/ölçekli saldırılar ZK'nın hiç kapsamadığı synthesis ağını da
bozar; koşullu saldırı ise tam ZK kapsamındaki `mapping.embed` üzerinde
olduğu halde yine geçerli ispat üretir.

## CUDA meselesi (kapandı) — hata verbatim

`--ops-impl auto` smoke-test'i `g_ema` GPU'ya taşındıktan (ve
`assert_module_on_device` geçtikten) sonra da başarısız oldu. Hata bir
cihaz sorunu değil GERÇEK derleme hatasıdır:

```
Setting up PyTorch plugin "bias_act_plugin"... Failed!
ModuleNotFoundError: No module named 'bias_act_plugin'
```

(ninja ve `CUDA_HOME=/usr/local/cuda` mevcut olduğu halde; Colab'ın
Python 3.13 / güncel PyTorch ortamında StyleGAN-XL custom op'ları
derlenemiyor.) Karar: `ref` modu KALICI; tüm sonuçlar `ops_impl='ref'`
ile üretildi (`ops_impl` ve `ops_impl_fallback_error` alanları JSON'da).

## Baseline ve güven aralığı altyapısı (hazırlandı, Colab'da HENÜZ koşulmadı)

- **`no_attack` baseline** (`ATTACK_NAMES[0]`): site'ın gerçek, değiştirilmemiş
  ağırlığı; `unprotected` = 4 dürüst siteyle normal FedAvg, `poisoned_alone`
  = temiz site tek başına. Aynı seed (0) ve `num_images_per_class=20` ile
  koşulunca tüm saldırı matrisleri bununla karşılaştırılabilir. Sonuç
  gelmeden "DR-4 istenince DR-0 üretiliyor" ifadesi KESİNLEŞMİŞ sayılmaz —
  temiz modelde de olabilir.
- **`--num-images-per-class N`** (varsayılan quick=20, full/custom=200) ve
  **`--seeds 0,1,2`**. Varsayılandan farklı N ve seed'ler ayrı sonuç anahtarı
  alır (`__n50`, `__seed1`); seed=0 + varsayılan N eski anahtarı korur, bu
  yüzden mevcut sonuçlar yeniden hesaplanmaz. FID/KID yalnızca ilk seed'de
  hesaplanır (seed'e bağlı değil, pahalı).
- **Seed'in neyi değiştirdiği:** üretilen z'ler ve KID alt-örneklemesi.
  Gerçek görüntü seti SABİT (sınıfın ilk N görüntüsü) — yani varyans
  tahmini gerçek-örnek seçiminden gelen belirsizliği KAPSAMAZ.
- Süre (0,498 s/görüntü, ref; yalnızca üretim): 20 görüntü/sınıf → 100
  görüntü ≈ 50 s/koşul-seed; 7 saldırı × 3 koşul = 21 koşul ≈ 17,5 dk/seed
  (yalnız baseline: 3 koşul ≈ 2,5 dk). 50 görüntü/sınıf → 250 görüntü ≈ 125 s;
  21 koşul ≈ 44 dk/seed, 3 seed ≈ 2,2 sa. Gerçek-görüntü yükleme ve Inception
  çıkarımı EK süredir (ölçülmedi).
- Sonuçlar birden çok seed için henüz otomatik özetlenmiyor (ortalama/std
  hesabı yok) — baseline görüldükten sonra kaç örnek/seed gerektiğine
  karar verilip eklenecek.

## Sınırlılıklar (makalede açıkça yer almalı)

1. **FID ölçülmedi.** `ref` modunda ölçülen 0,498 s/görüntü ile
   `fid50k_full`+`kid50k_full` koşul başına ≈13,8 saat; 18 koşul
   (6 saldırı × 3) ≈ 248 saat. FID'in kaba saldırılarda (random/scaled)
   ne gösterdiği ölçülmedi. Bu saldırılar zaten norm kapısınca
   yakalandığı için ana bulguyu etkilemez, ama FID'in bu saldırılardaki
   davranışı raporlanamaz.
2. **`--metrics-mode custom --fid-num-gen 5000` maliyet tahmini**
   (0,498 s/görüntü, ref; yalnızca üretim süresi — Inception çıkarımı ve
   ilk seferde gerçek-veri özellik önbelleği EK süredir):
   `run_custom_fid_kid` FID ve KID için ayrı ayrı 5000 görüntü üretir →
   koşul başına ≈ 2 × 5000 × 0,498 s ≈ 4 980 s ≈ **1,4 saat**; tam koşum
   18 koşul ≈ **25 saat**. Daha makul alt küme (koşul seçimi bayrağı
   olmadığından `--only-attack` ile saldırı bazında): {random_weights,
   scaled_poison_10x} için unprotected+protected, {conditional_poison,
   conditional_poison_compensated} için unprotected (protected aynı
   girdi) ≈ 6 koşul ≈ **8–9 saat**. Bu değerler 13,13 ile
   KARŞILAŞTIRILAMAZ (`fid_comparable_to_reference=False`), yalnızca bu
   koşumun iç karşılaştırması içindir.
3. **Saldırısız temel matris yok.** Temiz FedAvg için sınıf-tutarlılığı
   matrisi ölçülmedi; `perceived_class[4]=0` ve DR-4 satırındaki DR-0
   düşüklüğünün temiz modelde NE olduğu bilinmiyor. Kanıt, tek-başına↔
   FedAvg karşılaştırması ve satır-içi göreli farkla destekleniyor; bir
   `no_attack` temel çizgisi bu sınırlılığı kapatır.
4. **İstatistiksel güven.** 20 görüntü/sınıf, tek tohum (seed=0), KID için
   güven aralığı hesaplanmadı; matris değerleri (0,01–0,12) dar bir
   aralıkta ve DR sınıfları zaten görsel olarak yakın. Oranlar (1,27 /
   2,37 / 5,59 / 6,16) yön için güçlü kanıt, büyüklük için gösterge
   niteliğindedir.
5. **Tek round, tek zehirli site** (round 10, site 0); telafi formülü diğer
   dürüst sitelerin aynı embed satırlarına sahip olduğunu varsayar (hepsi
   aynı önceki global'den başlar).
6. **`ref` ↔ `cuda` eşdeğerliği doğrulanamadı** — aşağıya bakın.

## Sınırlılık: `13.13` ile karşılaştırılabilirlik (`ops_impl` moduna bağlı)

Faz A'daki mevcut `fid50k_full=13.13` ölçümü, ORİJİNAL eğitim
ortamında (CUDA custom op'ları BAŞARIYLA derlenmiş, `impl='cuda'`)
yapıldı. Bu Colab oturumunda CUDA derlemesi BAŞARISIZ OLDU
(`ops_impl='ref'`, kalıcı;, bkz. yukarıdaki "Tasarım kararları" madde 4) bu
fazın FID ölçümleri saf PyTorch REFERANS uygulamasıyla hesaplanıyor.
`bias_act`/`upfirdn2d`/`filtered_lrelu`'nun `ref` yolu MATEMATİKSEL
OLARAK `cuda` yoluyla AYNI sonucu üretmesi GEREKİR (StyleGAN-XL'in
kendi tasarım amacı budur — `cuda` sadece bir HIZ optimizasyonu) ama
bu eşdeğerlik BU PROJEDE doğrudan sayısal olarak DOĞRULANMADI (ör.
aynı ağırlıkla hem `cuda` hem `ref` modunda FID ölçüp fark
KIYASLANMADI — orijinal ortamda CUDA derlemesi başarısız olduğundan
böyle bir karşılaştırma bu oturumda YAPILAMADI). Bu yüzden:

- **Saldırı × koruma matrisinin İÇ karşılaştırmaları GEÇERLİ**: aynı
  koşumdaki TÜM hücreler (6 saldırı × 3 koşul) AYNI `ops_impl` ile
  ölçülüyor — göreli farklar (unprotected vs protected, saldırı A vs
  B) tutarlı bir zeminde karşılaştırılıyor.
- **`13.13` referansıyla DOĞRUDAN karşılaştırma, `ops_impl='ref'` ise
  DİKKAT gerektirir** — teorik olarak eşdeğer olması gereken ama bu
  oturumda sayısal olarak DOĞRULANAMAYAN bir varsayıma dayanıyor. Bu
  makalenin sınırlılıklar bölümüne AÇIKÇA yazılmalı; mümkünse ileride
  CUDA derlemesinin çalıştığı bir ortamda (`--ops-impl cuda` başarılı
  olursa) `unprotected`/`conditional_poison` YOK koşulunun FID'i
  ayrıca ölçülüp `13.13` ile DOĞRUDAN karşılaştırılarak bu varsayım
  kapatılabilir.


## Dürüstlük noktası (makalede aynen yer almalı)

Bu çalışmada norm kapısı, dışarıdan zorlanan ağır saldırıları (rastgele
ağırlık, ölçekli ΔG; ‖ΔG‖ tau'nun 8–84 katı) engelliyor — bunu norm
kapısı yapıyor, ZK değil. Meşru güncellemelerin doğal aralığında kalan,
sınıf-hedefli bir saldırıyı (‖ΔG‖ = 0,84·tau, FedAvg seyrelmesi telafi
edilse bile) ne norm kapısı ne ZK durduruyor; tespit için
davranışsal/istatistiksel analiz (sınıf-tutarlılığı karışıklık
matrisi) gerekiyor ve o da örnek küçüklüğü nedeniyle şu an yön kanıtı
düzeyinde. ZK'nın katkısı içerik denetimi değil, denetlenen ağırlığın
katkıda bulunulanla aynı olmasının garantisidir.
