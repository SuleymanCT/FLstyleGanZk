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
BİREBİR aynı ölçüm ayarlarıyla karşılaştırılabilir olması gerekiyor —
ama bkz. aşağıdaki "Sınırlılık: `13.13` ile karşılaştırılabilirlik".)

## Sınırlılık: `13.13` ile karşılaştırılabilirlik (`ops_impl` moduna bağlı)

Faz A'daki mevcut `fid50k_full=13.13` ölçümü, ORİJİNAL eğitim
ortamında (CUDA custom op'ları BAŞARIYLA derlenmiş, `impl='cuda'`)
yapıldı. Bu Colab oturumunda CUDA derlemesi BAŞARISIZ olduysa
(`ops_impl='ref'`, bkz. yukarıdaki "Tasarım kararları" madde 4) bu
fazın FID ölçümleri saf PyTorch REFERANS uygulamasıyla hesaplanıyor.
`bias_act`/`upfirdn2d`/`filtered_lrelu`'nun `ref` yolu MATEMATİKSEL
OLARAK `cuda` yoluyla AYNI sonucu üretmesi GEREKİR (StyleGAN-XL'in
kendi tasarım amacı budur — `cuda` sadece bir HIZ optimizasyonu) ama
bu eşdeğerlik BU PROJEDE doğrudan sayısal olarak DOĞRULANMADI (ör.
aynı ağırlıkla hem `cuda` hem `ref` modunda FID ölçüp fark
KIYASLANMADI — orijinal ortamda CUDA derlemesi başarısız olduğundan
böyle bir karşılaştırma bu oturumda YAPILAMADI). Bu yüzden:

- **Saldırı × koruma matrisinin İÇ karşılaştırmaları GEÇERLİ**: aynı
  koşumdaki TÜM hücreler (5 saldırı × 2 koşul) AYNI `ops_impl` ile
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
