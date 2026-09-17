# Faz E Zaman Araştırması — 6× Yavaşlık Neden Var?

**Durum: ÇÖZÜLDÜ (Colab'da doğrulandı) — optimize edilecek bir şey
YOK.** Bu doküman, Faz E'nin ilk gerçek ispatının (durum=success,
verified=True, tepe RAM 6,4 GB, deploy gas 3.281.404, verify gas
1.220.225, toplam gas 4.501.629) `total_ezkl_seconds=467,5` ölçmesinin
(Faz C3'ün AYNI yapılandırma — k=1, matmul, scale=8 — için ölçtüğü
`setup 37,3s + prove 37,4s ≈ 75s`'in **~6 katı**) NEDEN olduğunu
araştırdı.

## SONUÇ (Colab'da adım-adım ölçümle doğrulandı)

Aşağıdaki bölümlerdeki instrümantasyon eklendikten sonra alınan GERÇEK
bir ispatın adım-adım dökümü:

```
onnx_export        0.12s
gen_settings        0.50s
calibrate           1.83s
compile_circuit     0.06s
get_srs             0.42s
setup              36.48s
gen_witness         0.54s
prove              37.35s
verify_offchain     0.24s
```

**`get_srs` önbellekten geliyor (0.42s)** — madde 3(a)'daki "yeniden mi
indiriliyor" sorusu KESİN olarak yanıtlandı: HAYIR, önbellekleme
ÇALIŞIYOR. İlk koşudaki `467,5s` anomalisi, o TEK koşumda SRS'in İLK
KEZ indirilmesiydi (bir defalık maliyet) — sonraki tüm ispatlar
önbellekten okuyor. **Gerçek, tekrarlayan maliyet `setup + prove ≈ 74s`**
— Faz C3'ün referans ölçümüyle (`~75s`) TAM UYUMLU. **Optimize edilecek
bir şey yok**: kalibrasyon (1,83s) ve ONNX ihracı (0,12s) zaten ihmal
edilebilir düzeyde, tahmin edildiği gibi (madde 3b/3c). 60 ispatlık tam
koşunun gerçek toplam ezkl maliyeti ~74s × 60 ≈ 74 dakika (SRS'in tek
seferlik ilk indirme maliyeti hariç) — 8 saatlik ilk tahmin YANLIŞTI,
o tahmin 467,5s'lik ANOMALİYİ (bir kerelik SRS indirmesi) normal
davranış sanmaktan kaynaklanıyordu.

**Faz E'nin gerçek darboğazı ezkl DEĞİL, anvil'in bellek birikimiydi**
— 36 ispat sonrası anvil RPC'ye yanıt vermez oldu (`ReadTimeout`). Bu
AYRI bir altyapı sorunu, `docs/phase_e_infra_notes.md`'de ele alınıyor.

## 1. Kök sebep bulunamadı — çünkü adım-adım kırılım YOKTU

`total_ezkl_seconds` tek bir toplam sayıydı (`compute_mode_totals`'ın
`EZKL_TIMING_KEYS` üzerinden topladığı). Kod incelemesinde İKİ ayrı
sorun bulundu:

1. **`EZKL_TIMING_KEYS` sabiti EKSİKTİ** — `"get_srs"` ve `"onnx_export"`
   listede yoktu. `get_srs` süresi `scripts.bench_circuit.run_ezkl_pipeline`
   tarafından zaten HESAPLANIYOR ve `timings` sözlüğüne yazılıyordu, ama
   `replay_proofs.py`'nin toplam sayısına hiç KATILMIYORDU — yani gerçek
   toplam süre, raporlanan `467,5s`'ten bile daha büyük olabilir.
   `onnx_export` ise hiç ÖLÇÜLMÜYORDU (kod vardı, zamanlama yoktu).
2. **Adım bazında konsol çıktısı yoktu** — her ezkl adımı (`gen_settings`,
   `calibrate_settings`, `compile_circuit`, `get_srs`, `setup`,
   `gen_witness`, `prove`, `verify_offchain`) BAŞLAMADAN ÖNCE bir satır
   yazdırıyordu ama BİTİNCE geçen süreyi YAZDIRMIYORDU — süre sadece
   `result.json`'un içinde, gözden kaçacak şekilde duruyordu.

## 2. Eklenen ölçüm altyapısı (bu commit)

- `scripts/bench_circuit.py: run_ezkl_pipeline`/`run_prove_and_verify`:
  HER adımdan hemen sonra `"[bench_circuit]  <adım>: X.XXs"` satırı
  eklendi (`gen_settings`, `calibrate_settings`, `compile_circuit`,
  `get_srs`, `setup`, `gen_witness`, `prove`, `verify_offchain`).
- `orchestrator/round_runner.py: generate_and_submit_proof`: `export_to_onnx`
  çağrısı artık ZAMANLANIYOR (`onnx_export`, eskiden hiç ölçülmüyordu)
  ve fonksiyonun SONUNDA TÜM adımların (`onnx_export` + ezkl'nin 8
  adımı + `solc_compile`) toplu bir dökümü tek blokta yazdırılıyor
  (`=== round=R site=S adım-adım süre dökümü (toplam X.XXs) ===`).
- `scripts/replay_proofs.py: EZKL_TIMING_KEYS`: `"onnx_export"` ve
  `"get_srs"` eklendi — `total_ezkl_seconds` artık GERÇEKTEN ölçülen
  HER şeyi kapsıyor, sessizce eksik bırakmıyor.

Bir SONRAKİ Colab koşumu artık HANGİ adımın 6 katı büyüdüğünü doğrudan
konsolda ve `replay_results_<mod>.json`'un `timings` alanında gösterecek.

## 3. Kontrol edilmesi istenen üç hipotez — kod okumasıyla ön-değerlendirme

### (a) SRS her ispatta yeniden mi indiriliyor?

`circuits/ezkl_utils.py: run_get_srs(settings_path)` → `ezkl.get_srs(settings_path)`
— `srs_path` parametresi HİÇ VERİLMİYOR (`None` kalıyor). `ezkl.pyi`
(`get_srs(settings_path, logrows, srs_path)`) bunun opsiyonel olduğunu
gösteriyor ama ezkl'nin `srs_path=None` iken varsayılan olarak NEREYE
yazdığı/NEREDEN okuduğu (sabit bir global önbellek mi, yoksa her
seferinde `settings_path`'e GÖRELİ bir konum mu) kaynak koduyla
DOĞRULANAMADI (WebFetch ile GitHub kaynağı/discussion'ları tarandı,
kesin bir cevap bulunamadı — bkz. commit geçmişi). **Bu belirsiz
kalıyor** — ama artık `get_srs` HER ispatta AYRI ölçülüp raporlandığından
(madde 2), bir sonraki Colab koşumu bunu KESİN olarak gösterecek: eğer
`get_srs` HER ispatta tutarlı şekilde büyükse (ör. her seferinde
saniyeler/on saniyeler), bu güçlü bir "yeniden indiriliyor/üretiliyor"
kanıtı olur; küçük ve sabit kalırsa önbellekleme ÇALIŞIYOR demektir ve
darboğaz başka bir adımdadır.

### (b) Kalibrasyon her ispatta yeniden mi koşuyor?

EVET, ve bu BEKLENEN bir davranış — her (round,site)'ın ağırlıkları
FARKLI olduğundan (`param_visibility="fixed"`, Faz D'nin tasarım notu),
`calibrate_settings`'in kalibre ettiği scale/lookup parametreleri o
SPESİFİK ağırlık kümesine göre belirleniyor, ATLANAMAZ. Soru süresinin
Faz C3'ün referans koşumuna göre NE KADAR farklı olduğu — bu da madde
2'nin eklediği per-adım ölçümle netleşecek. Eğer `calibrate_settings`
tek başına 300+ saniye tutuyorsa, bu gerçek round/site ağırlıklarının
(Faz A'nın ZATEN eğitilmiş, potansiyel olarak Faz C3'ün C0 referansından
FARKLI değer dağılımına sahip) kalibrasyon aramasını daha uzun sürede
tamamlamasıyla açıklanabilir — ama bu bir VARSAYIM, ölçümle
doğrulanacak.

### (c) ONNX ihracı her ispatta mı yapılıyor?

EVET — her (round,site)'ın ağırlıkları FARKLI olduğundan, ONNX
grafiğinin (yapısı AYNI kalsa da) İÇİNDEKİ sabit ağırlık tensörleri
FARKLI, yani her ispat için YENİDEN ihraç edilmesi ZORUNLU (aynı
ONNX dosyasını ağırlık DEĞİŞTİRİLMEDEN yeniden kullanmak yanlış bir
devre ispatlar). Ama bu adımın (saf PyTorch→ONNX dönüşümü, ağ/disk
G/Ç'si yok) süresi muhtemelen ihmal edilebilir düzeyde (Faz C3'te
ayrıca ölçülmemişti çünkü zaten hızlı olduğu varsayılıyordu) — madde
2'nin eklediği ölçüm bunu da doğrulayacak.

## 4. Yeniden kullanılabilir olan / OLMAYAN — `param_visibility="fixed"` sınırı

| Bileşen | 60 ispat arasında PAYLAŞILABİLİR mi? | Gerekçe |
|---|---|---|
| ONNX grafiğinin YAPISI (op'lar/şekiller) | Yapısal olarak AYNI, ama dosya İÇERİĞİ (ağırlıklar) HER SEFERİNDE farklı — dosya paylaşılamaz, sadece "yeniden ihraç etmenin ucuz olduğu" bilgisi paylaşılır | Ağırlıklar ONNX'in İÇİNE gömülü (sabit tensör) |
| SRS (`get_srs`) | **EVET, PRENSİPTE paylaşılabilir** — SRS, devrenin `logrows`'una bağlıdır, AĞIRLIKLARA değil; k=1/matmul/scale=8 devresi HER round/site için AYNI `logrows`'u (Faz C3: 19) üretiyorsa, TEK bir SRS tüm 60 ispat için yeterli olmalı | KZG SRS evrenseldir, devre YAPISINA (logrows) bağlıdır, spesifik ağırlıklara değil |
| `settings.json`/kalibrasyon sonucu | HAYIR — ağırlıklara göre kalibre ediliyor (madde 3b) | Farklı ağırlık dağılımı → farklı optimal scale/lookup parametreleri (teorik olarak; PRATİKTE k/embed_mode/scale zaten SABİT olduğundan aynı `scale=8` sonucu çıkması BEKLENİR ama bu VARSAYILMAYACAK, ölçülecek) |
| `compile_circuit` çıktısı (derlenmiş devre) | HAYIR — `settings.json`'a (ve dolayısıyla ağırlıklara) bağlı | Aynı sebep |
| `setup` çıktısı (pk/vk) | **HAYIR, KESİNLİKLE** — Faz D'nin tasarım notu: `param_visibility="fixed"` ağırlıkları devrenin SABİT sütunlarına gömüyor, pk/vk BU SABİT DEĞERLERE göre üretiliyor | Bu yüzden zaten her (round,site) KENDİ Verifier'ını deploy ediyor |
| Verifier deploy/solc derleme | HAYIR — `setup`'a bağlı (yukarısı) | Aynı sebep |

**SONUÇ (Colab ölçümüyle KAPANDI):** `get_srs` DEĞİL darboğaz — zaten
önbellekten geliyor (0,42s, madde "SONUÇ" bölümü). Bu tablonun tek pratik
sonucu: sistemin GERÇEK, tekrarlayan maliyeti `setup + prove ≈ 74s`'dir
ve bu, `param_visibility="fixed"` (Faz D'nin tasarım notu) yüzünden
HİÇBİR şekilde paylaşılamaz/hızlandırılamaz — bu devre mimarisinin
DOĞAL, kabul edilmesi gereken maliyetidir. **Hiçbir optimizasyon
uygulanmadı, uygulanmasına gerek YOK.**

## Sıradaki adım

Zaman araştırması KAPANDI. Faz E'nin gerçek darboğazı anvil'in bellek
birikimiydi (60 ispatlık tam koşuda 36. ispattan sonra RPC yanıt
vermez oldu) — bu AYRI bir altyapı sorunu olarak `docs/phase_e_infra_notes.md`'de
ele alınıp segment-tabanlı bir anvil yeniden başlatma mimarisiyle
çözüldü (henüz Colab'da doğrulanmadı).
