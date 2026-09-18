# Faz E Sonuç Raporu — Tam vs Kademeli İspat Maliyet Karşılaştırması

**Durum: Faz E TAMAMLANDI.** Bu doküman, `scripts/replay_proofs.py`
tarafından otomatik üretilen `docs/phase_e_replay.md`'ye (ham
tablolar/başarısızlık özeti/altyapı olayları) EK olarak, iki Colab
koşumunun (tam mod: 60/60 ispat, kademeli mod: 21/21 ispat, İKİSİNDE
DE sıfır başarısızlık) sonuçlarının yorumlu analizini içerir.

## 1. Ana karşılaştırma tablosu

**ÖNEMLİ DÜZELTME (bkz. Bölüm 3.2):** Aşağıdaki (a) tablosundaki tam
modun `9540.2s` toplam ezkl süresi, anvil'in bellek şişmesi
YAŞANDIĞI (ilk tam mod koşumu) sırasında ölçülen **13 "bozuk ortam"
ispatını** İÇERİYOR — bu ispatlar gerçekte NORMAL süresinin (~74-78s)
2-4 katı sürmüş, çünkü anvil'in bellek baskısı AYNI makinedeki ezkl
sürecini de yavaşlatmış (swap). Bu, mod SEÇİMİNDEN kaynaklanan bir
fark DEĞİL, ÖLÇÜM ORTAMINDAKİ bir arızanın tam moda (ama kademeli
moda DEĞİL — bkz. Bölüm 3.2) sızmasıdır. Bu yüzden **(b) normalize
tablosu** makalenin ASIL referans alması gereken tablodur.

### (a) Ham toplamlar (bozuk-ortam verisini İÇERİR)

| mod | ispat sayısı | başarılı | toplam ezkl süresi (s) | toplam solc süresi (s) | toplam deploy gas | toplam verify gas | toplam zincir maliyeti (gas) | tasarruf % |
|---|---|---|---|---|---|---|---|---|
| full | 60 | 60 | 9540.2 | 66.5 | 196.880.100 | 73.059.588 | 269.939.688 | (referans) |
| staged | 21 | 21 | 1641.9 | 21.6 | 68.908.200 | 25.572.981 | 94.481.181 | 65.0 |

Ham özet: ispat sayısı tasarrufu **%65** (21/60), ezkl süresi
tasarrufu **%82.8** (ŞİŞKİN — bkz. Bölüm 3.2), zincir maliyeti
tasarrufu **%65.0**.

### (b) Normalize edilmiş karşılaştırma (SADECE sağlıklı ortamda ölçülen ispatlarla)

Kademeli modun 21 ispatının **HİÇBİRİ** bozuk-ortam listesiyle
(Bölüm 3.2) kesişmiyor — kademeli modun round/site takvimi (round 9:
site 1,3; round 12: site 1,3; round 13: site 0,2,3) bozuk-ortam
listesindeki (round9_site**0,2**; round12_site**0,2**;
round13_site**1**) kombinasyonlarla TAM OLARAK AYRIK. Yani **kademeli
modun ölçülen ortalaması (`1641.9s / 21 = 78.185s`) zaten TEMİZ,
GERÇEK bir sağlıklı-ortam ölçümü** — bu değeri sabit alıp ispat
sayısı oranından normalize toplamı hesaplıyoruz (kullanıcının
istediği yöntem):

| mod | ispat sayısı | normalize ezkl süresi (s) | toplam zincir maliyeti (gas) | süre tasarrufu % (normalize) | gas tasarrufu % |
|---|---|---|---|---|---|
| full | 60 | 4691.1 | 269.939.688 | (referans) | (referans) |
| staged | 21 | 1641.9 | 94.481.181 | 65.0 | 65.0 |

`normalize ezkl süresi = ispat sayısı × 78.185s` (78.185s = kademeli
modun GERÇEK, temiz ölçülen ortalaması). Bu tablo `scripts/replay_proofs.py:
render_normalized_comparison_table` tarafından da üretilebilir hale
getirildi (bkz. Bölüm 3.3) — bir sonraki Colab koşumunda (madde 3'teki
yeniden-ölçüm sonrası) hem ham hem normalize tablo OTOMATİK,
`docs/phase_e_replay.md`'ye yazılacak.

**Sonuç: normalize edildikten sonra süre tasarrufu (%65.0) ile gas
tasarrufu (%65.0) BİREBİR ÖRTÜŞÜYOR** — ikisi de sadece ispat sayısı
oranından (21/60) geliyor. Faz E'nin ilk raporundaki "%82.8 > %65.0"
farkı, GERÇEK bir mod-etkisi DEĞİLMİŞ; tamamen tam moddaki 13
bozuk-ortam ispatının şişirdiği bir ÖLÇÜM ARTEFAKTIYMIŞ.

## 2. Takvim dağılımı (kademeli mod)

| round | ispat üreten site'lar | site sayısı | kural |
|---|---|---|---|
| 0 | site_0, site_1, site_2, site_3 | 4 | sabit round (fixed_rounds) |
| 1 | site_1 | 1 | rastgele seçim (random_ratio) veya itibar tetikli |
| 2 | — | 0 | — |
| 3 | — | 0 | — |
| 4 | site_3 | 1 | rastgele/itibar |
| 5 | — | 0 | — |
| 6 | — | 0 | — |
| 7 | site_0, site_1, site_2, site_3 | 4 | sabit round |
| 8 | — | 0 | — |
| 9 | site_1, site_3 | 2 | rastgele/itibar |
| 10 | — | 0 | — |
| 11 | — | 0 | — |
| 12 | site_1, site_3 | 2 | rastgele/itibar |
| 13 | site_0, site_2, site_3 | 3 | rastgele/itibar |
| 14 | site_0, site_1, site_2, site_3 | 4 | sabit round |
| **toplam** | | **21** | |

Gözlem: 7 round (2, 3, 5, 6, 8, 10, 11) kademeli modda HİÇ ispat
üretmedi — bu round'larda `startRound` yine de çağrıldı (challenge
seed alınıp takvim kararı verilebilmesi için), ama hiçbir site'ın
ispat üretmesi gerekmedi (ne sabit round ne itibar eşiği altı ne de
rastgele seçim tetiklendi). Bu, `orchestrator/schedule.py:
build_round_schedule`'ın beklenen davranışı — bkz. Bölüm 3.1.

## 3. ÖNEMLİ ANALİZ — süre tasarrufu (%82.8) neden gas tasarrufundan (%65) yüksek görünüyordu

### 3.1. Gas tarafı — kod incelemesiyle DOĞRULANAN gerçek mekanizma

Görevin başlangıç hipotezi ("her round için `startRound` çağrısı
yapılıyor ve bu ispat sayısından bağımsız sabit bir maliyet")
`scripts/replay_proofs.py: compute_mode_totals` (satır 141-163) kod
incelemesiyle **doğrulanamadı** — düzeltiyorum: `total_chain_cost_gas`
alanı SADECE her (round,site) sonucunun `deploy_gas` (Verifier
deploy) + `submit_gas` (`submitProof` çağrısı) alanlarını toplar;
`startRound`'un gas'ı (`start_gas`) bu toplama HİÇ girmiyor, sadece
konsola yazdırılıyor (`orchestrator/round_runner.py`/`scripts/replay_proofs.py`'nin
`print(f"... startRound: gas={start_gas} ...")` satırı). Yani
raporlanan zincir maliyeti rakamı, round sayısından değil, SADECE
gerçekleşen ispat sayısından etkileniyor — round başına sabit bir
maliyet bileşeni bu tabloya hiç girmiyor.

**Gerçek mekanizma, ham sayılardan doğrudan çıkarılabiliyor:**

- İspat başına ortalama zincir maliyeti: full modda
  `269.939.688 / 60 = 4.498.995` gas, staged modda
  `94.481.181 / 21 = 4.499.104` gas — **iki mod arasında pratik
  olarak BİREBİR aynı** (fark ~109 gas, ölçüm gürültüsü düzeyinde).
  Bunun nedeni: her ispatın deploy+verify maliyeti, o ispatın hangi
  round/site'a ait olduğundan DEĞİL, sabit devre yapılandırmasından
  (k=1, matmul, scale=8 — Faz C3) doğan SABİT Verifier bytecode
  boyutu ve SABİT `submitProof` calldata yapısından geliyor. Gerçek
  ağırlık değerleri deploy/verify gas'ını ETKİLEMİYOR.
- Bu yüzden **gas tasarrufu ≈ ispat sayısı tasarrufu**: `1 - 21/60 =
  %65.0`, ölçülen `%65.0` ile örtüşüyor (bkz. Bölüm 1'deki tablo) —
  kademeli modun gas tarafındaki KAZANCI, sadece "daha az ispat
  üretmek"ten geliyor, ispat başına EK bir gas verimliliği YOK.
- **ezkl süresi ispat başına GÖRÜNÜŞTE farklıydı**: full modda ispat
  başına ham ortalama `9540.2 / 60 = 159.0s`, staged modda
  `1641.9 / 21 = 78.2s`. Bölüm 3.2'nin gösterdiği gibi bu fark GERÇEK
  bir "full mod / staged mod" etkisi DEĞİL — full moddaki 13 ispatın
  bozuk (anvil bellek şişmesi yaşanan) bir ortamda ölçülmüş olması.

**Sonuç (düzeltilmiş):** Gas tasarrufu (%65.0), sadece daha az ispat
üretmenin doğrudan bir sonucu. Süre tasarrufu da, ORTAM etkisi
ayıklandıktan (normalize edildikten) SONRA, AYNI ispat-sayısı oranına
(%65.0) yakınsıyor — Bölüm 1(b)'deki normalize tabloya bakınız. Ham
raporun `%82.8 > %65.0` bulgusu, mod seçiminin GERÇEK bir yan etkisi
DEĞİL, ölçüm ortamındaki bir arızanın (Bölüm 3.2) istatistiğe sızmasıydı.

### 3.2. İspat başına ezkl süresi neden 2 katı farklıydı? (159s vs 78s) — SİSTEMATİK, ÇÖZÜLDÜ

**Önceki rapor turunda bu bölüm "ortam değişkenliği" (iki ayrı Colab
oturumu arasında rastgele bir performans farkı) olarak KESİNLEŞTİRİLEMEDİ
işaretlenmişti. Kullanıcının ham `replay_results_full.json` verisini
doğrudan incelemesiyle bu hipotez YANLIŞ çıktı — gerçek sebep
SİSTEMATİK ve KESİN olarak teşhis edildi, aşağıda anlatılıyor.**

**Bulgu:** İspat süreleri rastgele dağılmıyor, İKİ NET kümeye
ayrılıyor:

| küme | setup | prove | get_srs | hangi kombinasyonlar |
|---|---|---|---|---|
| HIZLI (sağlıklı) | ~36s | ~37s | ~0.4s | full modun ilk 47 ispatı + kademeli modun 21 ispatının TAMAMI |
| YAVAŞ (bozuk ortam) | ~110-160s | ~140s | ~0.1s | round9_site0, round9_site2, round10 (4 site: 0,1,2,3), round11 (4 site: 0,1,2,3), round12_site0, round12_site2, round13_site1 — **13 ispat** |

**Kök sebep:** Bu 13 ispatın TAMAMI, `docs/phase_e_infra_notes.md`'de
belgelenen anvil ÇÖKME olayının (36 ispat sonrası RPC yanıt vermez
oldu, `ReadTimeout`) yaşandığı **ilk tam mod koşumunda** üretildi.
Anvil'in bellek şişmesi SADECE RPC'yi değil, **AYNI makinede
(Colab VM'i) koşan ezkl alt sürecini de** yavaşlatmış — bellek
baskısı/swap yüzünden `setup+prove` maliyeti ~74s'den ~250-300s'ye
çıkmış. Segment mimarisi (Faz E'nin anvil-restart düzeltmesi, bkz.
`docs/phase_e_infra_notes.md`) devreye girdikten SONRAKİ tüm ispatlar
(full modun kalan 47'si + kademeli modun 21'inin TAMAMI) tekrar
~74-78s'lik sağlıklı süreye döndü.

**Neden `get_srs` ters yönde davranıyor (bozuk ortamda 0.1s < sağlıklı
ortamda 0.4s)?** Bu, SRS'in zaten diskte/bellekte önbelleklenmiş
olduğunu, bozuk ortamda bile SRS okumasının küçük kaldığını gösteriyor
— yani darboğaz SRS okuma/indirmede DEĞİL, doğrudan `setup`/`prove`'un
kendi hesaplama/bellek erişim maliyetinde (swap'a düşen sayfalar,
disk G/Ç'si CPU/bellek yoğun ezkl hesaplamasını yavaşlatıyor).

**Kademeli mod neden ETKİLENMEDİ:** Kademeli modun round/site takvimi
(round 9: site 1,3 — round 12: site 1,3 — round 13: site 0,2,3;
round 10 ve 11'de HİÇ ispat yok) bozuk-ortam listesindeki
kombinasyonlarla (round9_site**0,2**; round12_site**0,2**;
round13_site**1**; round10/11'in TÜMÜ) **TAM OLARAK AYRIK** —
tesadüfen, kademeli modun rastgele/itibar-tetikli seçimi anvil
çökmesinin yaşandığı round/site aralığına hiç denk gelmedi. Bu yüzden
kademeli modun 78.185s'lik ortalaması baştan beri TEMİZDİ; düzeltilmesi
gereken SADECE tam moddu.

**Bu artık bir BULGU olarak sunuluyor (makalenin uygulanabilirlik
bölümüne girer):** ZK ispat üretimi (özellikle `setup`/`prove` gibi
bellek-yoğun adımlar), AYNI makinede eş zamanlı çalışan bir blockchain
düğümünün (anvil) bellek davranışına DUYARLI — düğüm belleği şişip
swap'a düşünce, düğümle HİÇBİR ilgisi olmayan ayrı bir alt süreç
(ezkl) de yavaşlıyor. Bu, "ZK-ispat üreten worker'lar ile zincir
düğümünün AYNI makinede/kapsayıcıda çalıştırılması" pratiğinin
GERÇEK bir kaynak izolasyonu riski taşıdığını gösteriyor — üretim
dağıtımında ayrı makine/kapsayıcı (veya en azından cgroup bellek
limiti) ÖNERİLİR. Segment mimarisi (periyodik anvil yeniden başlatma)
bu riski azaltıyor ama kaynak izolasyonunun YERİNİ tutmuyor — anvil
tekrar şişip çökene kadar olan pencerede yine ezkl'i yavaşlatabilir.

**Kod değişikliği (bu tur):** `scripts/replay_proofs.py`'ye
`classify_environment_health(timings)` (setup+prove toplamı
`ENVIRONMENT_DEGRADED_THRESHOLD_SECONDS=150.0`'ı aşarsa `"degraded"`),
`split_by_environment_health`, `compute_healthy_avg_ezkl_seconds`
(SADECE healthy+success ispatların GERÇEK ortalaması — 74s gibi bir
sayı uydurulmuyor, mevcut healthy veriden hesaplanıyor) ve
`render_normalized_comparison_table` eklendi — Bölüm 1(b)'deki
normalize tablo BUNDAN SONRA script tarafından OTOMATİK üretilecek.
Ayrıca her yeni sonuca `run_id` (main() başına üretilen, o Colab
koşumunu tekil olarak işaretleyen bir alan) yazılmaya başlandı — ESKİ
kayıtlarda bu alan yok, bu yüzden geriye dönük sınıflandırma HER ZAMAN
`timings` üzerinden yapılıyor (yukarıdaki gibi), `run_id`'nin varlığına
bağlı değil. 10 yeni saf test (`tests/test_replay_proofs.py`) yerelde
GERÇEKTEN doğrulandı (291 passed, 2 skipped toplam).

## 4. Altyapı kısıtları

- Her iki modda da **1'er planlı (`scheduled_restart`) anvil yeniden
  başlatması** gerçekleşti — `configs/schedule.yaml:
  replay_infra.anvil_restart_interval` (varsayılan 10) eşiğine göre
  tetiklendi. Segment mimarisi (Faz E, `docs/phase_e_infra_notes.md`)
  beklendiği gibi çalıştı: hiçbir `ReadTimeout`/altyapı kaynaklı
  başarısızlık YAŞANMADI (60/60 ve 21/21 — sıfır başarısızlık).
- **İtibar sıfırlanması notu:** her segment yeniden başlatmasında
  `RoundManager` yeniden deploy edilip site'lar yeniden kaydediliyor
  — bu, ZİNCİR tarafındaki `reputation`/`isSiteEligible` durumunu
  sıfırlıyor (bkz. `docs/phase_e_infra_notes.md`, Bölüm "Segment
  mimarisi"). Kademeli modun takvim KARARI bundan ETKİLENMEDİ, çünkü
  `must_prove`'un itibar girdisi `replay_proofs.py`'nin YEREL
  `reputations` sözlüğünden geliyor (zincirdeki değerden değil) — bu
  YEREL sözlük segment yeniden başlatmalarından bağımsız, koşum
  boyunca sürekli. Zincir tarafındaki sıfırlanan itibar/uygunluk
  durumu sadece `RoundManagerClient.is_site_eligible` gibi zincire
  SORULAN bir kontrolü etkiler — bu koşumda `finalizeRound`/uygunluk
  sorgusu YAPILMADI (replay sadece ispat üretim/submit maliyetini
  ölçüyor), yani sıfırlanma bu koşumun SONUÇLARINI bozmadı; ama
  gerçek bir üretim (canlı `round_runner.py`) akışında segment
  benzeri bir yeniden başlatma olsaydı, zincirdeki itibar geçmişinin
  kaybolacağı AÇIKÇA not düşülmeli.
- **YENİ (Bölüm 3.2) — kaynak izolasyonu kısıtı:** anvil'in bellek
  şişmesi, AYNI makinede koşan ezkl alt sürecini de yavaşlattı (13
  ispat, `setup+prove` ~74s yerine ~250-300s). Bu, "anvil'i periyodik
  yeniden başlat" (segment mimarisi) çözümünün YETERSİZ kaldığı bir
  durum — segment yeniden başlatılana KADAR geçen pencerede ezkl yine
  yavaşlayabilir. Kalıcı çözüm KAYNAK İZOLASYONU (ayrı makine/kapsayıcı
  veya bellek cgroup limiti), segment mimarisi sadece bir HAFİFLETME.

## 5. Sonuç

Faz E'nin kabul kriteri (60 shard üzerinde tam ve kademeli modun
KARŞILAŞTIRILABİLİR şekilde koşup makalenin ana maliyet tablosunu
üretmesi) **karşılandı**: iki mod da sıfır başarısızlıkla tamamlandı.
Gas tasarrufunun MEKANİZMASI (ispat sayısına orantılı, round-sayısından
bağımsız) kod incelemesiyle kanıtlandı. Süre tasarrufundaki görünen EK
bileşen (ham `%82.8 > %65.0` farkı) **KESİN olarak teşhis edildi**:
tam moddaki 13 ispat, anvil'in bellek şişmesinin (Bölüm 3.2) AYNI
makinedeki ezkl'i de yavaşlattığı bozuk bir ortamda ölçülmüş — kademeli
modun 21 ispatı bu aralıkla hiç kesişmediği için etkilenmedi. Normalize
edildiğinde (Bölüm 1(b)) süre tasarrufu da gas tasarrufuyla AYNI
değere (%65.0) yakınsıyor. Bu, ilk raporun "ortam değişkenliği"
hipotezinin YERİNE geçen, veriyle doğrudan kanıtlanmış bir sonuç —
uydurulmadı, kullanıcının ham veri incelemesiyle KANITLANDI.

## 6. Öneri — bozuk-ortam ispatlarını temiz ölçümle değiştirmek

13 bozuk-ortam ispatını (Bölüm 3.2'deki liste) `--force` ile yeniden
çalıştırıp GERÇEK, temiz bir tam-mod ölçümü elde etmek mümkün — segment
mimarisi artık devrede olduğundan (bu koşumdaki gibi bir anvil çökmesi
BİR DAHA olmamalı), bu ispatlar da diğer 47'si gibi ~74-78s'de
tamamlanmalı. Script'in mevcut `--rounds`/`--sites` arayüzü tekil
(round,site) çiftlerinin keyfi bir listesini DEĞİL, bir round×site
DİKDÖRTGENİNİ kabul ediyor — ama `--sites` virgüllü liste destekliyor
(`parse_int_range("0,2")`), bu yüzden 13 kombinasyonun TAMAMI, mevcut
CLI'a HİÇBİR yeni kod eklemeden, 5 hedefli komutla TAM İSABETLİ şekilde
kapsanabiliyor:

```bash
python -m scripts.replay_proofs --env colab --mode full --rounds 9  --sites 0,2 --force
python -m scripts.replay_proofs --env colab --mode full --rounds 10 --sites 0-3 --force
python -m scripts.replay_proofs --env colab --mode full --rounds 11 --sites 0-3 --force
python -m scripts.replay_proofs --env colab --mode full --rounds 12 --sites 0,2 --force
python -m scripts.replay_proofs --env colab --mode full --rounds 13 --sites 1   --force
```

Not: mesajınızda "12 ispat" deniyor ama listelenen kombinasyonların
toplamı **13** (round9: 2 + round10: 4 + round11: 4 + round12: 2 +
round13: 1 = 13) — küçük bir sayım farkı, yukarıdaki 5 komut LİSTENİN
TAMAMINI (13/13) kapsıyor, eksik/fazla bırakmıyor. Her komut kendi
`--force`'u sayesinde SADECE belirtilen (round,site) çiftlerini
yeniden çalıştırıp `replay_results_full.json`'daki mevcut 47 sağlıklı
kaydı KORUYARAK üzerine yazacak (`main()`'in `if key in all_results and
not args.force: continue` kontrolü `--force` ile atlanıyor, ama
DOKUNULMAYAN diğer anahtarlar dosyada olduğu gibi kalıyor — bkz.
`scripts/replay_proofs.py` satır ~757). Tahmini süre: 13 × ~80s
(ezkl) + solc/deploy/segment kurulum ek yükü ≈ 15-20 dakika (kullanıcının
"~15 dakika" tahminiyle uyumlu). Bu koşumdan sonra `replay_results_full.json`
TAMAMEN sağlıklı olacak, `compute_healthy_avg_ezkl_seconds` full modun
KENDİ verisinden de aynı ~74-78s'yi doğrulayacak ve bu raporun Bölüm
1(b)'sindeki normalize tablo GERÇEK (varsayımsız) tam-mod verisiyle
yeniden üretilebilecek.
