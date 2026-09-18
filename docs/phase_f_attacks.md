# Faz F — Saldırı × Koruma Matrisi (FID/KID/Sınıf-Tutarlılığı)

**Durum: quick mod Colab'da GEÇTİ (10/10) — norm/ZK sonuçları GERÇEK
ve KRİTİK bir bulgu içeriyor (aşağıya bakın). FID/KID hâlâ
KOŞULMADI** (`ref` modunda `--metrics-mode full` ~138 saat sürdüğü
için pratik değildi — bkz. "Sınırlılık" bölümü ve yeni `--metrics-mode
custom`). Bu doküman `{zk_root}/attacks/attack_results.json`'dan
doldurulan bir rapordur — CLAUDE.md madde 6 gereği hiçbir sayı
UYDURULMAZ; norm/ZK sütunları kullanıcının paylaştığı GERÇEK konsol
çıktısından, FID/sınıf-tutarlılığı sütunları ise `attack_results.json`
dosyasının kendisi PAYLAŞILDIĞINDA doldurulacak (bkz. dosya sonundaki
"Bekleyen veri" notu).

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
**Bu düzeltmeyle CUDA'nın GERÇEKTEN derlenip derlenmediği henüz
YENİDEN test EDİLMEDİ** — bir sonraki Colab koşumu bunu gösterecek;
çalışırsa `full` modu saatler yerine dakikalar sürer.

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

## ANA TABLO — norm/ZK sütunları GERÇEK (quick mod koşumu, 10/10 geçti); FID/KID/sınıf-tutarlılığı PENDING

`delta_norm`/`norm_caught` GERÇEK Colab ölçümü (kullanıcı tarafından
raporlandı, `--metrics-mode quick`). `zk_in_scope`/`zk_caught` KOD
GARANTİSİ (`attacks/detection.py: verify_commitment_consistency`
aynı state'i kendisiyle karşılaştırıyor — hash eşitliği MATEMATİKSEL
olarak KESİN, Colab'a özgü bir belirsizlik YOK, bu yüzden "beklenen"
değil "kesin" olarak işaretleniyor). FID/KID/sınıf-tutarlılığı
sütunları `{zk_root}/attacks/attack_results.json`'un GERÇEK içeriği
OLMADAN doldurulamıyor — quick mod class_confusion'ı GERÇEKTEN
hesaplıyor ama bu doküman o JSON'u henüz OKUMADI (bu oturumda dosyaya
erişim yok, sadece konsol logundaki özet rakamlar paylaşıldı).
**Sonraki adım: `attack_results.json`'un içeriği (en azından her 10
sonucun `class_confusion` alanı) paylaşılırsa bu tablo GERÇEK
sayılarla tamamlanır — bkz. dosya sonundaki "Bekleyen veri" notu.**

| saldırı | koşul | \|\|ΔG\|\| | tau=3000 aşıldı mı (norm) | ZK kapsamda mı | ZK yakaladı mı | FID | sınıf tutarlılığı | zehirli site dahil mi |
|---|---|---|---|---|---|---|---|---|
| random_weights | unprotected | 59592 | **EVET** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | evet (kapısız) |
| random_weights | protected | 59592 | **EVET** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | HAYIR (norm dışladı) |
| scaled_poison_10x | unprotected | 25066 | **EVET** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | evet (kapısız) |
| scaled_poison_10x | protected | 25066 | **EVET** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | HAYIR (norm dışladı) |
| scaled_poison_50x | unprotected | 125332 | **EVET** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | evet (kapısız) |
| scaled_poison_50x | protected | 125332 | **EVET** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | HAYIR (norm dışladı) |
| scaled_poison_100x | unprotected | 250664 | **EVET** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | evet (kapısız) |
| scaled_poison_100x | protected | 250664 | **EVET** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | HAYIR (norm dışladı) |
| **conditional_poison** | unprotected | **2507** | **HAYIR** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | evet (kapısız) |
| **conditional_poison** | protected | **2507** | **HAYIR** | evet | HAYIR (kod garantisi) | (JSON gerekli) | (JSON gerekli) | **EVET (HİÇBİR mekanizma dışlamadı)** |

**KRİTİK SONUÇ (zaten kesinleşti, JSON beklemez):** `conditional_poison`
protected koşulda bile zehirli siteyi DIŞLAMIYOR — `||ΔG||=2507 <
tau=3000` olduğundan norm kontrolü geçiyor, ZK zaten tasarım gereği
hiçbir zaman yakalamıyor (madde 2). Bu, makalenin ANA TEZİ: mevcut
sistemin norm+ZK savunma katmanı, meşru güncellemelerin doğal
aralığında saklanan bir sınıf-hedefli saldırıya karşı KÖRDÜR — tau
KEYFİ değil (Faz A'nın gerçek p99=2518 ölçümünden), ama bu saldırı
TAM OLARAK o meşru aralığın İÇİNDE kalacak şekilde TASARLANABİLİYOR.
Tespitin TEK yolu davranışsal/istatistiksel analiz (sınıf-tutarlılığı
matrisi) — bkz. aşağıdaki "FID vs sınıf-tutarlılığı: hangisi neyi
gösteriyor".

(Referans: mevcut deneyin korumasız FID'i `13.13` — `fid50k_full`,
BİREBİR aynı ölçüm ayarlarıyla karşılaştırılabilir olması gerekiyor —
ama bkz. aşağıdaki "Sınırlılık: `13.13` ile karşılaştırılabilirlik".)

## FID vs sınıf-tutarlılığı: hangisi neyi gösteriyor (item 4 — raporun çerçevesi)

Bu iki metrik BİLEREK FARKLI saldırı sınıflarını yakalamak için var —
biri diğerinin YERİNE geçmiyor:

- **FID (`fid50k_full`/`--metrics-mode custom`)**: KABA saldırıları
  gösterir. `random_weights`/`scaled_poison_50x`/`100x` gibi
  ağırlıkları GÖRÜNÜR şekilde bozan saldırılar, ÜRETILEN GÖRÜNTÜLERİN
  GENEL KALİTESİNİ/gerçekçiliğini düşürerek FID'i YÜKSELTİR —
  ama bu saldırılar zaten norm kontrolüyle YAKALANIYOR (yukarıdaki
  tablo), yani FID'in bunları göstermesi PRATİKTE gerekmiyor bile
  (savunma zaten önce devreye giriyor).
- **Sınıf-tutarlılığı matrisi (`class_confusion_matrix`)**: İNCE,
  HEDEFLİ saldırıyı gösterir. `conditional_poison` görüntü KALİTESİNİ
  bozmuyor (FID'i muhtemelen DEĞİŞTİRMEZ — görüntüler hâlâ gerçekçi
  fundus görüntüleri) ama ÜRETİLEN GÖRÜNTÜNÜN SINIF KİMLİĞİNİ
  bozuyor — bu SADECE sınıf-bazlı bir karşılaştırmayla (üretilen DR-0
  görüntüsü GERÇEKTE DR-4'e mi benziyor?) görülebilir, tek bir
  toplam kalite skoruyla (FID) GÖRÜNMEZ.

**Rapor bu çerçeveye göre kuruldu**: yukarıdaki ANA TABLO'nun ilk dört
saldırısı için "hangi mekanizma yakaladı" sorusunun cevabı zaten norm
sütunundan OKUNUYOR (FID'e gerek KALMADAN); `conditional_poison`
satırları için asıl kanıt sınıf-tutarlılığı matrisinin DR-0/DR-4
karışıklığı gösterip göstermediği — bu, `attack_results.json`
okunduğunda KESİNLEŞECEK.

## `conditional_poison`'ın sınıf-tutarlılığı matrisi GELDİ — takas GÖRÜNMÜYOR (yeni sorun, araştırılıyor)

Kullanıcının paylaştığı GERÇEK matris (unprotected/protected — Bölüm
"ANA TABLO"nun gösterdiği gibi BİREBİR AYNI, çünkü koruma bu saldırıyı
zaten durduramadı):

| istenen sınıf | en yakın gerçek sınıf (argmin KID) | KID değeri | beklenen (takas varsayımıyla) |
|---|---|---|---|
| DR-0 | DR-0 | 0.0489 | DR-4 |
| DR-4 | DR-1 | 0.0599 | DR-0 |

`swap_detected` DR-2 ve DR-4 için `true` dönüyor ama bunlar
UYGULANAN takasla (DR-0↔DR-4) İLGİSİZ — köşegen BÜYÜK ÖLÇÜDE
korunmuş. **Takas, üretilen görüntülerin sınıf kimliğinde
GÖRÜNMÜYOR.** Üç hipotez, SIRAYLA test ediliyor:

- **H1 — FedAvg seyreltmesi**: zehirli site 4 siteden biri, takaslanmış
  gömme sadece 1/4 ağırlıkla ortalamaya giriyor, diğer 3 site DOĞRU
  gömmeyi getiriyor — ortalama sonuç orijinal gömmeye YAKIN kalmış
  olabilir.
- **H2 — Gömme takası yetersiz**: sınıf bilgisi SADECE `embed`
  satırında değil, `embed_proj`/`fc0`/`fc1` üzerinden de akıyor
  olabilir — takas `w` vektörünü GERÇEKTEN değiştirmiyor olabilir.
- **H3 — KID ayrım gücü yetersiz**: 20 görüntü/sınıf (quick mod) az,
  DR sınıfları görsel olarak zaten YAKIN (matris değerleri dar bir
  `0.04-0.11` aralığında) — ayrım GÜCÜ yetersiz kalıyor olabilir.

### H1 testi — ŞİMDİ eklendi, Colab'da koşulmayı bekliyor

`scripts/run_attacks.py`'ye YENİ bir tanı koşulu eklendi:
**`poisoned_alone`** (`CONDITIONS` artık `("unprotected", "protected",
"poisoned_alone")`) — `build_poisoned_alone_global` FedAvg'ı TAMAMEN
ATLAYIP zehirli site'ın KENDİ ağırlığını (seyreltme YOK, ağırlık 1.0)
doğrudan "global" olarak kullanıp AYNI `evaluate_condition` (FID +
sınıf-tutarlılığı) işlem hattından geçiriyor. Bu koşul TÜM saldırılar
için otomatik hesaplanıyor (sadece `conditional_poison` için özel bir
dal AÇILMADI — aynı mekanizma her saldırı için tutarlı bir
karşılaştırma tabanı sağlıyor).

**Yorumlama kılavuzu:**
- `poisoned_alone`'da takas (DR-0→DR-4, DR-4→DR-0) GÖRÜNÜYORSA → **H1
  DOĞRU** (seyreltme sorunu) — saldırı GÜÇLENDİRİLMELİ (aşağıdaki
  öneriye bakın).
- `poisoned_alone`'da DA görünmüyorsa → **H1 ELENDİ**, sıradaki adım
  H2 testi (aynı `z` ile takas öncesi/sonrası `w` çıktısını
  karşılaştırmak — kod HENÜZ yazılmadı, kullanıcının isteği üzerine
  ÖNCE H1 sonucu bekleniyor).

### Eğer H1 doğrulanırsa (seyreltme confirmed): önerilen güçlendirme

Kullanıcının sorduğu "hangi yolu önerirsin" sorusuna cevap: **basit
takas yerine, ortalama sonucun HEDEFE ulaşacağı şekilde
KOMPANSE EDİLMİŞ bir gömme değeri öneriyorum** — birden fazla
kötücül site EKLEMEK yerine (bu, TEHDİT MODELİNİ "tek kötücül site"
yerine "koordineli çoklu kötücül site"ye DEĞİŞTİRİR, ki bu ayrı ve
daha güçlü bir varsayım gerektirir; `scaled_poison`'ın zaten kapsadığı
"tek site ne kadar agresif olabilir" sorusuna da karşılık gelmez).
Gerekçe: `scaled_poison`, ΔG'yi ÖLÇEKLEYEREK aynı seyreltme sorununu
ZATEN çözdü (FedAvg'ın 1/4 ağırlığını telafi etmek için saldırıyı
BÜYÜTTÜ) — AYNI mantık `conditional_poison`'a da uygulanabilir:
basit satır takası yerine, zehirli site'ın KATKISI

```
poisoned_row = 4 * target_row - 3 * honest_row
```

olarak hesaplanmalı — burada `target_row` takas hedefi (ör. DR-4'ün
gerçek gömme satırı), `honest_row` DR-0'ın (değiştirilmeden önceki)
gerçek gömme satırı. `(3*honest_row + poisoned_row)/4 = target_row`
eşitliğini SAĞLAYACAK şekilde çözülmüş — yani FedAvg SONRASI ortalama
TAM OLARAK hedef değere ulaşıyor, basit takasın aksine (ki o zaten
1/4'e seyreliyordu). Bu, `attacks/conditional_poison.py`'ye
`swap_embed_rows`'un YANINA (onun YERİNE değil — basit takas kendi
başına da GEÇERLİ bir "en kaba koşullu saldırı" senaryosu, ayrı
tutulmalı) yeni bir `compensated_swap_embed_rows(state_dict, row_a,
row_b, num_sites, *, embed_key)` fonksiyonu olarak eklenebilir —
`num_sites` FedAvg'daki toplam site sayısı (bu projede 4). **Bu kod
HENÜZ yazılmadı** — H1 sonucunu bekliyor, kullanıcının "sonuca göre
devam ederiz" talimatına uygun olarak.

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

## Doğrulanan desen (quick mod, `--metrics-mode quick`, 10/10 koşul geçti)

- **`random_weights`/`scaled_poison_10x`/`50x`/`100x`: `||ΔG||`
  (59592/25066/125332/250664) `tau=3000`'i AÇIKÇA AŞIYOR — norm
  kontrolü YAKALIYOR, zehirli site protected koşulda dışlanıyor
  (DOĞRULANDI, artık "beklenen" değil).** `scaled_poison_10x`'in bile
  (en küçük ölçek) `tau`'nun ~8 katı bir norm ürettiği ÖNEMLİ — bu
  saldırı ailesi norm kontrolüne karşı hiç "sınırda" değil, RAHATÇA
  yakalanıyor.
- **`conditional_poison`: `||ΔG||=2507 < tau=3000` — norm kontrolü
  KAÇIRIYOR (DOĞRULANDI).** Faz A'nın gerçek p99=2518 ölçümünden
  seçilen `tau`'nun HEMEN ÜZERİNDE kalıyor — yani bu saldırı,
  norm eşiğinin keyfi OLMAMASINA RAĞMEN, meşru güncellemelerin doğal
  aralığında SAKLANABİLİYOR.
- **`zk_caught` BEŞ saldırının BEŞİ İÇİN DE `False` — bu Colab
  ölçümüne bağlı DEĞİL, kod garantisi** (`verify_commitment_consistency`
  bir state'i kendisiyle karşılaştırıyor, hash eşitliği matematiksel
  olarak KESİN). "ZK içerik değil tutarlılık kanıtlar" bulgusunun
  doğrudan kanıtı — Colab'da ayrıca DOĞRULANMASI beklenmiyor, zaten
  kesin.
- **PENDING (JSON gerekli):** sınıf-tutarlılığı matrisinde
  `perceived_class[0]==4`/`perceived_class[4]==0` (DR-0/DR-4 swap)
  görünüyor mu? Bu, `conditional_poison`'ın GERÇEKTEN tespit
  edilebilir olduğunun kanıtı olacak — `attack_results.json`'un
  `conditional_poison__unprotected`/`conditional_poison__protected`
  anahtarlarının `class_confusion` alanı paylaşılırsa KESİNLEŞİR.

## Dürüstlük noktası (makalede aynen yer almalı)

ZK doğrulamasının bu sistemdeki GERÇEK rolü — taahhüt-ispat-katkı
tutarlılığı, İÇERİK/kalite denetimi DEĞİL — GERİLİMLİ ama ÖNEMLİ bir
sınırlılık: zincir üzeri ZK katmanı TEK BAŞINA zehirlemeye karşı
yeterli DEĞİL, istatistiksel/davranışsal analiz (FID/KID/sınıf-
tutarlılığı gibi) TAMAMLAYICI bir savunma katmanı olarak GEREKLİ. Bu
ayrım makalenin "ZK'nın sınırlılıkları" bölümüne AÇIKÇA girmeli —
ZK'yı olduğundan güçlü göstermemek adına.

## Bekleyen veri

`conditional_poison`'ın unprotected/protected sınıf-tutarlılığı
ÖZETİ artık GERÇEK (yukarıdaki "takas GÖRÜNMÜYOR" bölümü) — ama şunlar
HÂLÂ bekliyor:

1. **H1 testinin sonucu**: `poisoned_alone` koşulunun
   `conditional_poison__poisoned_alone` kaydının `class_confusion`
   alanı — takas ORADA görünüyor mu? (kod yazıldı, Colab'da HENÜZ
   koşulmadı).
2. Diğer 4 saldırının (`random_weights`/`scaled_poison_*`) TAM
   `class_confusion` matrisleri — bu saldırıların norm kontrolüyle
   zaten yakalandığı biliniyor ama sınıf-tutarlılığı üzerindeki
   yan etkileri (ör. görüntü kalitesi bu kadar bozulunca sınıf
   ayrımı da rastgeleleşiyor mu) henüz görülmedi.
3. 5×5 tam matrislerin TAMAMI (sadece DR-0/DR-4 satırları değil) — 
   `swap_detected`'ın DR-2/DR-4'te neden `true` döndüğünü (uygulanan
   takasla İLGİSİZ) anlamak için.
4. (Madde 7'nin `--metrics-mode custom` koşumu ayrıca yapılırsa) FID
   sütunu — `13.13` ile karşılaştırılamaz notuyla birlikte.

`attack_results.json` dosyasının TAMAMI (ya da en azından
`conditional_poison__poisoned_alone`'un `class_confusion` alanı)
paylaşıldığında H1/H2/H3 kesinleştirilip rapor tamamlanacak.
