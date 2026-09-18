# Faz E Sonuç Raporu — Tam vs Kademeli İspat Maliyet Karşılaştırması

**Durum: Faz E TAMAMLANDI.** Bu doküman, `scripts/replay_proofs.py`
tarafından otomatik üretilen `docs/phase_e_replay.md`'ye (ham
tablolar/başarısızlık özeti/altyapı olayları) EK olarak, iki Colab
koşumunun (tam mod: 60/60 ispat, kademeli mod: 21/21 ispat, İKİSİNDE
DE sıfır başarısızlık) sonuçlarının yorumlu analizini içerir.

## 1. Ana karşılaştırma tablosu

| mod | ispat sayısı | başarılı | toplam ezkl süresi (s) | toplam solc süresi (s) | toplam deploy gas | toplam verify gas | toplam zincir maliyeti (gas) | tasarruf % |
|---|---|---|---|---|---|---|---|---|
| full | 60 | 60 | 9540.2 | 66.5 | 196.880.100 | 73.059.588 | 269.939.688 | (referans) |
| staged | 21 | 21 | 1641.9 | 21.6 | 68.908.200 | 25.572.981 | 94.481.181 | 65.0 |

Ek özet: ispat sayısı tasarrufu **%65** (21/60), ezkl süresi tasarrufu
**%82.8**, zincir maliyeti tasarrufu **%65.0** (gas rakamı ispat
sayısı tasarrufuyla ayrık ondalığa kadar örtüşüyor — bkz. Bölüm 3).

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

## 3. ÖNEMLİ ANALİZ — süre tasarrufu (%82.8) neden gas tasarrufundan (%65) yüksek

### 3.1. Kod incelemesiyle DOĞRULANAN gerçek mekanizma

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
- **ezkl süresi ise ispat başına da değişiyor**: full modda ispat
  başına ortalama `9540.2 / 60 = 159.0s`, staged modda
  `1641.9 / 21 = 78.2s` — staged modun ispat başına süresi full'un
  **YARISI**. Bu, `calibrate_settings`'in `param_visibility="fixed"`
  yüzünden GERÇEK ağırlık değerlerine göre veri-bağımlı bir arama
  yaptığını doğrulayan Faz E'nin zaman araştırmasıyla (`docs/phase_e_timing_investigation.md`)
  tutarlı — ama tam olarak NEDEN 2 katı fark olduğu Bölüm 3.2'de ayrı
  ele alınıyor (kesin kanıtlanamadı, dürüstçe belirtiliyor).

**Sonuç:** Gas tasarrufu, sadece daha az ispat üretmenin doğrudan bir
sonucu (çarpan etkisi yok); süre tasarrufu HEM daha az ispat ÜRETMENİN
HEM DE üretilen her ispatın (ortalamada) daha KISA sürmesinin bileşik
etkisi — bu yüzden `%82.8 > %65.0`.

### 3.2. İspat başına ezkl süresi neden 2 katı farklı? (159s vs 78s)

Kontrol edilenler:
- **Segment/anvil yeniden başlatma sayısı EŞİT** (her iki modda da
  1'er planlı yeniden başlatma, bkz. Bölüm 4) — bu, segment
  mimarisinin kendisinin (yeniden deploy, yeniden kayıt) iki mod
  arasında farklı bir ek yük yaratmadığını gösteriyor; segment yapısı
  TEK BAŞINA bu farkı açıklamıyor.
- **SRS önbellekleme**: `docs/phase_e_timing_investigation.md`'in
  hâlâ AÇIK bıraktığı bir soru — `ezkl.get_srs(settings_path)`
  `srs_path=None` iken TAM OLARAK nereye yazıp nereden okuduğu
  kaynak koduyla kesin doğrulanamamıştı. Bu, olası bir KISMİ katkı
  olabilir (ör. iki koşum arasında disk önbelleğinin farklı davranması)
  — ama gözlenen farkın 60 örnek üzerinde TUTARLI ve TEMİZ bir ~2×
  oranı olması (birkaç ara sıra sıçrayan aykırı değer değil), bunun
  TEK başına yeterli bir açıklama olma ihtimalini zayıflatıyor: SRS
  yeniden indirme (bir kerelik ~467s'lik bilinen anomali) rastgele/
  seyrek olay olsaydı ortalamaya düzensiz sıçramalar olarak yansırdı,
  DÜZGÜN bir 2× çarpanı olarak değil.
- **En olası açıklama (KESİNLEŞTİRİLEMEDİ, dürüstçe işaretleniyor):**
  iki mod AYRI Colab oturumlarında koşturuldu (tam mod ~2,65 saat
  sürekli ezkl hesaplaması, kademeli mod ~27 dakika) — Colab'ın
  paylaşımlı/değişken VM tahsisi, disk G/Ç baskısı (60 kez ~2,5GB'lık
  pk dosyası yazıp silme vs 21 kez) veya oturum düzeyinde başka bir
  ortam farkı, AYNI ezkl koduna AYNI türde iş yükü verilse bile
  ölçülen süreyi sistematik olarak etkileyebilir. Bu, mod SEÇİMİNİN
  kendisinden değil, İKİ AYRI koşumun ortam koşullarından kaynaklanan
  bir varyans olabilir.
- **BİLİNÇLİ TEST SINIRI / veri boşluğu:** bu raporun dayandığı veri
  sadece her modun TOPLU (`compute_mode_totals`) rakamları — hangi
  round/site'ın hangi ADIMDA (calibrate/setup/prove/get_srs) ne kadar
  sürdüğünü gösteren ham `replay_results_full.json`/
  `replay_results_staged.json` bu oturumda incelenmedi (Colab/Drive'da,
  yerel makinede değil). Kesin kök sebep belirlemek için ÖNERİLEN
  somut bir sonraki adım: round 0, 7, 14 HER İKİ modda da ispatlandı
  (sabit round'lar) — aynı gerçek (round,site) ağırlıklarının İKİ
  AYRI koşumdaki per-adım sürelerini doğrudan karşılaştırmak, mod
  farkını mı yoksa ortam farkını mı gördüğümüzü kesin olarak ayırt
  ederdi. Bu karşılaştırma bu raporda YAPILMADI (ham dosyalar
  mevcut değildi) — ileride ham JSON'lar elde edilirse tek satırlık
  bir script ile yapılabilir.

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

## 5. Sonuç

Faz E'nin kabul kriteri (60 shard üzerinde tam ve kademeli modun
KARŞILAŞTIRILABİLİR şekilde koşup makalenin ana maliyet tablosunu
üretmesi) **karşılandı**: iki mod da sıfır başarısızlıkla tamamlandı,
gas tasarrufunun MEKANİZMASI (ispat sayısına orantılı, round-sayısından
bağımsız) kod incelemesiyle kanıtlandı, süre tasarrufundaki EK
bileşen (ispat başına süre farkı) tespit edildi ama kesin kök sebebi
mevcut veriyle KANITLANAMADI — bu açıkça bir sınır olarak işaretlendi,
uydurulmadı.
