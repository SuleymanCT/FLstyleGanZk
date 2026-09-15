# PLAN.md

Altı fazın tanımı ve kabul kriterleri. Her fazın durumu, o faz
tamamlandığında güncellenir.

## Ortam politikası

Faz B ve sonrası TAMAMEN Colab'da çalışır (Windows'ta foundry/anvil
kurulumu sancılı olduğundan bu karar değişti). Yerel makine sadece kod
yazımı ve `pytest` birim testleri için kullanılır — `ezkl` proving,
`anvil`/`forge`, gerçek zincir/web3 işlemleri hiç yerelde çalışmaz
(bkz. `CLAUDE.md` madde 7). `scripts/setup_local.sh` bu araçları artık
kurmuyor; onları gerektiren testler (`tests/test_toy_pipeline.py` gibi)
yerelde otomatik **skip** olur, Colab'da gerçekten koşar
(`notebooks/colab_runner.ipynb`, `MODE="toy"`).

## Ortak altyapı: pathguard

`storage/pathguard.py` (`assert_writable`, `open_readonly`) tüm
fazları bağlar: Drive'daki korumalı köklere (`Generative_Image`,
`FL_Experiments`, `parallel_final`, `StyleGANTrain`) yazma artık kod
seviyesinde engellenir. **Bundan sonra dosya yazan HER yeni script**
(Faz C'nin ONNX/devre çıktıları, Faz D'nin kontrat/orkestratör
çıktıları, Faz E'nin replay sonuçları, Faz F'nin saldırı/canlı koşu
çıktıları dahil) her yazma çağrısından önce `assert_writable` çağırmak
ZORUNDA — bkz. `CLAUDE.md` madde 9. Faz A'daki üç script
(`scripts/inventory.py`, `extract_shards.py`, `audit_fedavg.py`) buna
göre retrofit edildi.

## Faz A (Colab) — Envanter ve çıkarma

**durum: tamamlandı**
Colab'da gerçek Drive verisiyle koşuldu: `inventory.json` (60/60
(round,site) çifti eksiksiz), `delta_norms.json` (56 delta, p50/p90/p99
çıkarıldı) ve `fedavg_audit.json` (15 round'un tamamında numaralandırma
doğrulandı: `fedavg_N.pt` = round_N'in dört sitesinin fp32 ortalaması)
üretildi ve `zk_artifacts_results/`'a commit edildi. Ayrıntılı analiz
(mapping katmanları, fedavg numaralandırma tutarlılığı, delta-G normu
tablosu, round 14 anomalisi, mimari tutarlılık) `docs/phase_a_report.md`'de.
Kod: `scripts/inventory.py`, `scripts/extract_shards.py`,
`scripts/audit_fedavg.py`, `fl/module_tree.py`, `fl/shard_utils.py`,
`fl/fedavg_utils.py`, `fl/inventory_utils.py`, `fl/stylegan_xl_env.py`
— hepsi her yazmadan önce `storage.pathguard.assert_writable` çağırıyor,
pkl/fedavg okumaları `storage.pathguard.open_readonly` ile yapılıyor
(bkz. "Ortak altyapı" notu yukarıda).

**Çözülmüş nokta:** `docs/phase_a_report.md` Bölüm D'de detaylandırılan
round 14 anomalisi (4 sitenin delta-G normu ~0.0001 farkla neredeyse
birebir aynı) `scripts/probe_duplicates.py` ile netleştirildi —
`zk_artifacts_results/duplicate_check.json`, 15 round'un tamamında 4
benzersiz shard hash'i ve hiçbir site-global eşleşmesi olmadığını
gösteriyor. Kopya/kayıt hatası ihtimali ELENDİ: dört site gerçekten
farklı ağırlıklara yakınsıyor, sadece IID bölme + eşit tick sayısı
nedeniyle norm cinsinden benzer mesafe kat ediyorlar (ayrıntı: rapor
Bölüm D). `configs/schedule.yaml: tau_norm_threshold` bu ölçüme göre
3000.0 olarak dolduruldu.

Girdi yapısı:
```
parallel_final/round_0..round_14/site_0..site_3/00000-stylegan3-r-site*/
    network-snapshot.pkl  (StyleGAN-XL sınıf pickle'ı, düz state_dict
    DEĞİL — dnnlib ve torch_utils sys.path'te olmalı, legacy.load_network_pkl
    ile açılır), log.txt, stats.jsonl, training_options.json
parallel_final/fedavg_0.pt ... fedavg_N.pt
```

- **A1** `scripts/inventory.py` — round/site tamlığı, her pkl'ın
  anahtarları, G_ema'nın modül ağacı (mapping ve embedding alt modül
  isimleri, parametre sayıları, z_dim/c_dim/w_dim/num_layers — AYRINTILI
  yazdır), training_options tutarlılığı, stats.jsonl'dan FID'ler, base
  model araması. Çıktı: `inventory.json`.
- **A2** `scripts/extract_shards.py` — her (round, site) için G_ema'dan
  sadece mapping + sınıf gömme ağırlıklarını fp32 `.pt` shard olarak
  çıkar (~MB'lar), yanına `meta.json` (tam G'nin canonical_hash'i,
  parametre sayıları, tensör şekilleri). Her pkl'ı işledikten sonra
  belleği serbest bırak. `progress.json` ile devam edebilir olsun
  (Colab kopar).
- **A3** `scripts/audit_fedavg.py` — `fedavg_N.pt` gerçekten round_N'in
  dört sitesinin fp32 ortalaması mı? `fedavg_(N+1)` ile de karşılaştır
  ve numaralandırmanın yönünü kesinleştir. Ayrıca ΔG normlarını
  hesapla, p50/p90/p99 çıkar (tau eşiği bundan seçilecek).
  Karşılaştırmayı katman katman streaming yap, tam G'yi belleğe alma.

**Kabul:** `inventory.json` + 60 shard + `delta_norms.json` üretildi,
fedavg numaralandırması kesinleşti.

## Faz B (Colab) — Oyuncak uçtan uca

**durum: tamamlandı**
(kod yazıldı — bkz. `circuits/toy_model.py`, `circuits/ezkl_utils.py`,
`circuits/toy_pipeline.py`, `chain/anvil.py`, `chain/client.py`,
`tests/test_toy_pipeline.py`. ezkl çağrı sırası (gen_settings →
calibrate_settings → compile_circuit → get_srs → setup → gen_witness →
prove → verify) `tez-projesi/src/zk.py`'deki Colab'da KANITLANMIŞ
sırayla birebir aynı. EVM kısmı (`create_evm_verifier`,
`encode_evm_calldata`, anvil deploy, web3 doğrulama) resmi ezkl 23.0.5
Python binding dokümantasyonuna göre yazıldı ama `ezkl`/`anvil`/`solc`
yerelde yok — kabul kriteri ancak Colab'da tam koşumdan sonra
karşılanabilir. Saf/testable kısımlar (`ToyMLP`, `chain/anvil.py`'nin
metin ayrıştırma fonksiyonları, `circuits/ezkl_utils.run_async`)
yerelde test edildi, `tests/test_toy_pipeline.py` ezkl/anvil/solc
yokken otomatik skip olur.

**İlk Colab koşumu (1. tur):** ZK adımlarının hepsi geçti (setup 9.1s,
prove 13.5s, offchain verify 0.04s). Tek hata `generate_solidity_verifier`da
— `ezkl.create_evm_verifier` de (`get_srs` gibi) coroutine döndürüyor
ama senkron çağrılmıştı ("RuntimeError: no running event loop").
Düzeltildi: `circuits/ezkl_utils.run_async` genelleştirildi (artık tek
bir ezkl fonksiyonuna özel değil, HERHANGİ bir ezkl çağrısını —
sonucunun `inspect.isawaitable` olup olmadığına çalışma zamanında
bakarak — event loop içinde çalıştırıyor), `toy_pipeline.py`'deki TÜM
ezkl çağrıları (`gen_settings`, `calibrate_settings`, `compile_circuit`,
`setup`, `gen_witness`, `prove`, `verify`, `create_evm_verifier`,
`encode_evm_calldata`) bunun üzerinden geçiyor.

**2. tur:** `generate_solidity_verifier` artık geçiyor (0.847s). Yeni
hata `compile_verifier_solidity`de — ezkl'nin ürettiği Verifier.sol
yoğun assembly içerdiğinden varsayılan solc ayarlarıyla "Stack too
deep" hatası verdi. Düzeltildi: `chain/solc.py` (yeni, Faz D'nin
`RoundManager.sol` derlemesinde de kullanılacak) — `solc --standard-json`
ile `viaIR=true`, `optimizer.enabled=true`/`runs=200`,
`evmVersion="shanghai"` ayarlarıyla derliyor; hata mesajlarını
(`formattedMessage`) okunabilir basıyor (sadece returncode'a bakmıyor);
`timeout=300s` ile aşırı uzayan derlemeyi net hatayla durduruyor;
deployed (runtime) bytecode boyutunu EIP-170'in 24576 byte sınırına
karşı kontrol edip aşarsa uyarı basıyor ve raporda `exceeds_eip170`
olarak taşıyor. Saf ayrıştırma mantığı (`_parse_standard_json_output`)
sentetik solc çıktılarıyla test edildi.

**3. tur:** viaIR açıldı ama FARKLI bir hatayla karşılaşıldı: `YulException:
Cannot swap Variable usr$l_blind with Slot TMP[mulmod, 0]: too deep in
the stack by 1 slots` — yani tek bir (viaIR, runs) kombinasyonu her
zaman yetmiyor. Düzeltildi: `chain/solc.py: compile_with_fallback_strategies`
eklendi — `DEFAULT_SOLC_VERSIONS` (`0.8.20`, `0.8.24`, `0.8.26`) ×
`DEFAULT_STRATEGIES` (`viaIR=True` ile `runs∈{1,50,200}`, sonra
`viaIR=False` ile `runs=200`) kombinasyonlarını SIRAYLA dener, her
denemenin sonucunu loglar, ilk başarılıyı `used_strategy` etiketiyle
döner, hepsi başarısız olursa hepsinin özetini içeren tek bir hata
fırlatır. `read_source_header` Verifier.sol'un ilk 20 satırını (pragma)
loglar. Farklı solc sürümüne geçiş `solc-select install/use` ile
yapılıyor (`_ensure_solc_version`). SON ÇARE olarak
`deploy_and_verify_via_ezkl_native` eklendi: tüm kombinasyonlar
başarısız olursa, ezkl'nin resmi Python binding'inde doğrulanmış kendi
`deploy_evm`/`verify_evm` fonksiyonları (sol dosyasını doğrudan alıp
kendi iç solc çağrısıyla derleyip deploy/doğrulama yapıyor) devreye
giriyor; bu yolda `deploy_gas`/`verify_gas` receipt'ten doğrudan
alınamadığından `chain.client.Web3Client.find_transaction_gas` ile
zincir geriye taranarak kurtarılmaya çalışılıyor (bulunamazsa `None`,
uydurulmaz).

**4. tur — BAŞARILI, Faz B TAMAMLANDI:** `chain.solc.compile_with_fallback_strategies`
`solc=0.8.20, viaIR=False, optimizer_runs=200` ile (4. denemede)
derledi — native fallback'e (`deploy_and_verify_via_ezkl_native`) hiç
gerek kalmadı. Zincirin TAMAMI gerçekten çalıştı, ispat gerçekten
zincire gönderilip doğrulandı. Ölçülen sayılar (ayrıntılı analiz:
`docs/phase_b_report.md`):

| Adım | Süre |
|---|---:|
| export_onnx | 0.178s |
| ezkl_setup | 11.173s |
| ezkl_prove | 13.566s |
| ezkl_verify_offchain | 0.042s |
| generate_solidity_verifier | 0.191s |
| compile_verifier_solidity | 20.515s (4 denemenin toplamı) |
| deploy_and_verify_onchain | 0.511s |

`deploy_gas=2.956.287`, `verify_gas=546.938`,
`deployed_bytecode_size=13.426` byte (EIP-170 sınırı 24.576'nın
%54,6'sı). `DEFAULT_STRATEGIES` sırası, çalışan kombinasyon
(`viaIR=False, runs=200`) ilk sıraya gelecek şekilde güncellendi —
sonraki koşularda 3 gereksiz başarısız deneme yapılmayacak.

(Artık "yerel" değil "Colab" fazı — bkz. yukarıdaki "Ortam politikası"
notu.) Küçük MLP (32->32->8) -> ONNX -> ezkl derleme+kalibrasyon ->
ispat -> Solidity verifier üretimi -> anvil'de deploy -> Python'dan
doğrulama. `tests/test_toy_pipeline.py` tek komutla yeşil geçsin, adım
süreleri ve gas yazsın. anvil'i test içinden başlat ve kapat.

**Kabul:** test yeşil, boru hattı çalışıyor — KARŞILANDI. Buradan sonra
sahte çıktı yok.

## Faz C (yerel + Colab) — Gerçek mapping devresi

**durum: yapılmadı**

- **C0** `scripts/make_reference_outputs.py` — kod yazıldı (yeni:
  `fl/reference_utils.py` (saf, yerelde test edildi: `build_class_indices`,
  `build_z_c`, `resolve_mapping_kwargs`) + ince orkestrasyon script'i.
  `round_14/site_0`'ın (varsayılan, `--round`/`--site` ile değişebilir)
  gerçek `G_ema.mapping`'ini sabit tohumdan üretilen k∈{1,4,8} adet
  (z,c) ile çalıştırıp `w`'yi `{zk_root}/reference/mapping_ref_k{k}.pt`
  içine `{z,c,w,seed,source_pkl,shard_hash}` olarak kaydediyor.
  `shard_hash`, TAM G'nin değil sadece mapping+embedding shard'ının
  `canonical_hash`'i (C1'in karşılaştıracağı doğru kimlik).
  `G_ema.mapping.forward`'ın `inspect.signature`'ı loglanıyor,
  `truncation_psi`/`truncation_cutoff` sadece imzada GERÇEKTEN varsa
  gönderiliyor (varsayım yok). Boyutlar (z_dim/c_dim/w_dim/num_ws) ve
  mapping/embedding prefixleri inventory.json'dan değil CANLI G_ema'dan
  okunuyor — script self-contained, Faz A1'in ayrıca koşmuş olmasına
  gerek yok. `ezkl`/StyleGAN-XL yerelde yok, Colab'da HENÜZ koşulmadı —
  gerçek `w` şekli (beklenen `(k, num_ws, w_dim)`) ve mapping çağrı
  imzası ancak o koşumdan sonra kesinleşir.)

  **C0 sonucu (Colab'da koştu):** `G_ema.mapping.forward` imzası
  `(z, c, truncation_psi=1.0, truncation_cutoff=None, update_emas=False)`;
  `w` şekli `(k, 16, 512)`; `z_dim=64, c_dim=5, w_dim=512`; shard
  prefixleri `mapping, mapping.embed, mapping.embed_proj, mapping.fc0,
  mapping.fc1`; `round_14/site_0` shard_hash'i `duplicate_check.json`'daki
  aynı site hash'iyle birebir uyuştu (çapraz doğrulama). Referans
  dosyaları hazır: `mapping_ref_k{1,4,8}.pt`. Ayrıca
  `make_reference_outputs.py`'ye geriye uyumlu bir ek yapıldı: artık
  `embed_proj`/`fc0`/`fc1` çıktılarını `register_forward_hook` ile
  yakalayıp `"intermediates"` alanına da kaydediyor (C1'in katman-bazlı
  teşhisi için) — mevcut k1/k4/k8 dosyalarında bu alan yok, yeniden
  koşulursa eklenir.

- **C1** `circuits/rebuild_mapping.py` — kod yazıldı. StyleGAN-XL'e
  HİÇ bağımlı değil (`dnnlib`/`torch_utils`/`persistence` import
  etmiyor, grep ile doğrulandı) — sadece `torch.nn`. Matematik
  `autonomousvision/stylegan-xl`'in GERÇEK kaynak kodundan (WebFetch ile
  verbatim çekildi: `networks_stylegan3.py: FullyConnectedLayer` +
  `MappingNetwork`, `bias_act.py: activation_funcs['lrelu']`) alındı —
  equalized-lr `weight_gain`/`bias_gain` ölçeklemesi, `lrelu` (alpha=0.2,
  gain=sqrt(2)) doğru uygulanıyor; `embed_proj` (lr_multiplier=1.0, fc0/
  fc1'den FARKLI) ile `fc0`/`fc1` (lr_multiplier=0.01) net ayrıştırıldı.
  Çıkan mimari sabitleri (66048/262656/20544/320000 parametre) Faz A'nın
  GERÇEK inventory sayılarıyla birebir örtüşüyor (bağımsız çapraz
  doğrulama). Modül `(k, w_dim)` döner (`(k, num_ws, w_dim)` DEĞİL);
  truncation/update_emas/broadcast dahil edilmedi (psi=1.0'da devre
  dışı). `tests/test_rebuild_mapping.py`: sentetik ağırlıklarla, formül
  `rebuild_mapping`'in kodu çağrılmadan BAĞIMSIZ elle hesaplanıp
  `atol=1e-6` ile eşleştiği doğrulandı (11 test, hepsi yeşil) — gerçek
  StyleGAN-XL'e ulaşmadan güçlü bir doğruluk kanıtı, ama KESİN kanıt
  yine de Colab'daki 1e-5 karşılaştırması.

  `circuits/compare_utils.py` (yeni, saf) ve `scripts/verify_rebuild.py`
  (yeni, ince orkestrasyon) de yazıldı: shard+referansı yükler,
  `shard_hash` çapraz kontrolü yapar (yanlış çift karşılaştırmayı
  önler), referanstaki 16 `num_ws` kopyasının birebir aynı olduğunu
  doğrular, `max_abs_diff/mean_abs_diff/cosine_similarity` raporlar,
  eşik (1e-5) aşılırsa katman-bazlı teşhis dener. Sentetik shard+referans
  ile UÇTAN UCA yerel smoke testi yapıldı (shard_hash eşleşti,
  max_abs_diff=0.0) — ama bu sentetik ağırlıklarladır, gerçek StyleGAN-XL
  ağırlıklarıyla `ezkl`/Colab'da HENÜZ koşulmadı.

- **C2** `circuits/export_mapping.py` — kod yazıldı VE yerelde
  GERÇEKTEN uçtan uca test edildi (`onnx`/`onnxruntime` StyleGAN-XL/
  ezkl/anvil gibi Colab'a özgü değil, sade CPU paketleri — bu oturumda
  `pip install onnx onnxruntime` ile yerel test ortamına kuruldu; Faz
  B/C0/C1'in "sadece Colab'da doğrulanabilir" sınırının aksine, bu
  fazda export+onnxruntime karşılaştırmasının TAMAMI sentetik
  ağırlıklarla yerelde koştu — `max_abs_diff=0.0` her adımda).

  **Önce embed budaması** (`circuits/rebuild_mapping.py`'ye eklendi):
  kullanıcının bulgusu — `embed` 1000×320 (320.000 parametre, mapping
  toplamının ~yarısı) StyleGAN-XL'in ImageNet varsayılanından kalma,
  bizim `c_dim=5` ile 995 satır hiç kullanılmıyor. `verify_prunable_embed_rows`
  GERÇEK `c` tensörlerinin (C0'ın referans dosyalarından) `argmax`'ını
  hesaplayıp `[0,5)` dışında bir indeks var mı diye DOĞRULUYOR (varsaymıyor);
  `build_pruned_mapping_network` embed'i `[:5]`'e budayıp
  `build_mapping_network_from_shard`'ı (değişmeden) yeniden kullanıyor.
  Budanmış/budanmamış çıktılar aynı (z,c) üzerinde karşılaştırılıp
  `max_abs_diff==0.0` doğrulanmadan ONNX ihracına GEÇİLMİYOR (sentetik
  testte + smoke testte doğrulandı).

  ONNX ihracı: `dynamic_axes` YOK (k sabit, ezkl'nin istediği gibi),
  `opset` argümanla konfigüre edilebilir (varsayılan 13). `assert_num_ws_copies_identical`
  `scripts/verify_rebuild.py`'den `circuits/compare_utils.py`'ye taşındı
  (DRY, ikisi de kullanıyor).

  **ÖNEMLİ BULGU (yerel testte doğrulandı, ezkl uyumluluğu Colab'da
  netleşecek):** ihraç edilen ONNX grafiğinde `ArgMax` VE `Gather`
  düğümleri var (`c.argmax(dim=1)` + embedding lookup'tan) — bkz. C3
  notu aşağıda, ezkl bu op'ları desteklemeyebilir.

- **C3** `circuits/calibrate.py` + `scripts/bench_circuit.py` — ızgara:
  k ∈ {1,4,8} × bit ∈ {8,16}. Her kombinasyon için kısıt sayısı,
  derleme/setup/ispat/doğrulama süresi, tepe RAM, ispat boyutu.
  Patlayan kombinasyon "başarısız" işaretlenir, koşu düşmez.
- **C4** Kuantizasyon etkisi: w_q ve fp32 w arasında kosinüs benzerliği
  ve L2 farkı, DR sınıfı başına ayrı.

**Kabul:** en az bir (k, bit) kombinasyonu 30 dk altında ispat
üretiyor.

**Faz C2'den not (C3 için risk):** İhraç edilen ONNX grafiğinde
`ArgMax` ve `Gather` düğümleri var (`c.argmax(dim=1)` ile sınıf
indeksini bulup `embed`'den satır çekmekten geliyor — bkz. C2).
ezkl'nin bu op'ları destekleyip desteklemediği HENÜZ bilinmiyor; C3'ün
ilk adımı `gen_settings`/`compile_circuit`'in bu iki op'la ne yaptığını
görmek olmalı. Desteklenmiyorsa olası çözüm: `c.argmax` + `Gather`
yerine `c @ embed.weight[:5]` (matmul ile "seçim") — one-hot `c` zaten
elde, bu matematiksel olarak birebir aynı sonucu verir ama ArgMax/Gather
yerine tek bir MatMul kullanır (ZK devrelerinde yaygın bir "seçim"
tekniği) — ama bu bir mimari değişiklik olacağından şimdiden
uygulanmadı, sadece ezkl gerçekten tıkanırsa gündeme gelecek.

**Faz B'den not (C3/C4 için girdi):** Oyuncak MLP'nin (32→32→8, ezkl
varsayılan `calibrate_settings(..., "resources")` ile otomatik
scale/logrows seçimi — muhtemelen `input_scale=13`, `param_scale=13`)
ilk Colab koşumunda `gen_witness`/`prove` sırasında "decomposition
error: integer ... is too large" tipi UYARILAR görüldü (fatal değil —
Numerical Fidelity Report `max_abs_error=0.00033` ile ispat yine de
geçti). Bunlar muhtemelen bazı ara toplamların (Linear katmanının
32 terimlik iç çarpımı gibi) o scale'de seçilen decomposition/lookup
aralığını an be an aşıp ezkl'in otomatik olarak birden fazla parçaya
("leg") bölmesinden kaynaklanıyor — fonksiyonel bir hata değil ama
sınıra ne kadar yakın çalışıldığının işareti. Gerçek mapping ağı
(`mapping.fc1`: 512 terimlik iç çarpımlar, ~16x daha fazla terim)
aynı scale'de bu sınıra DAHA yakın ya da onu AŞAN davranış gösterebilir.
C3'ün (k, bit) taramasında bu uyarıların sıklığı/varlığı ve
`max_abs_error`'un k/bit'e göre nasıl değiştiği de kısıt sayısı/süre
yanında AYRI bir sütun olarak kaydedilmeli — düşük `bit_width` (8)
gerçek ağda bu uyarıları fatal hataya (ya da sessiz hassasiyet kaybına)
çevirebilir, `bit_width=16` daha güvenli başlangıç noktası olabilir.

**Faz B'den ikinci not (C3 için ek sütun — bytecode boyutu):** Oyuncak
MLP'nin verifier kontratı 13.426 byte (EIP-170'in 24.576 byte sınırının
%54,6'sı) — bkz. `docs/phase_b_report.md`. Gerçek mapping devresi
(669.248 parametre, oyuncağın ~627 katı) çok daha fazla kısıt
içereceğinden verifier boyutu da büyüyecek; Halo2/ezkl verifier boyutu
kısıt sayısıyla kabaca doğrusal büyüdüğünden EIP-170'i aşma ihtimali
gerçek. Bu yüzden **C3'ün (k, bit) benchmark ızgarasına
`deployed_bytecode_size` ve `exceeds_eip170` de birer sütun olarak
eklenmeli** (`chain.solc.compile_with_fallback_strategies`'in zaten
döndürdüğü alanlar) — ispat/doğrulama başarılı olsa bile verifier
deploy edilemiyorsa (EIP-170 aşılıyorsa) o (k, bit) kombinasyonu da
"başarısız" sayılmalı.

## Faz D (yerel) — Kontratlar ve orkestratör

**durum: yapılmadı**

`contracts/RoundManager.sol`: `registerSite`, `startRound(roundId,
globalCID, globalHash, challengeSeed)`, `submitUpdate(roundId,
updateCID, commitment)`, `submitProof -> Verifier.verifyProof`,
`finalizeRound(includedSites[])`, itibar mapping'i + eşik altında
dışlama, her adımda event. `Verifier.sol`'u ezkl üretir, elle yazma.

`chain/client.py`: web3.py sarmalayıcı, her çağrıda gas ölç.

`orchestrator/`: round_runner (challenge üretimi seed'den
deterministik), schedule (kademeli vs tam mod), aggregate (norm
kontrolü, NaN/Inf kontrolü, fp32 FedAvg, sadece zincirde onaylananlar).

`tests/test_contracts.py`: mutlu yol, geçersiz ispat reddi, itibar
düşüşü, çift submit. GERÇEK ezkl ispatlarıyla test et, sahte bytes ile
değil.

**Kabul:** testler yeşil, gas rakamları raporlandı.

## Faz E (yerel + Colab) — Replay

**durum: yapılmadı**

`scripts/replay_proofs.py`: mevcut 15 round shard'ları üzerinde, hiç
eğitim yapmadan ispat takvimini koştur. İki mod: tam (her round her
site) ve kademeli. Süre, boyut, doğrulama, gas karşılaştırması.

**Kabul:** maliyet karşılaştırma tablosu üretildi.

## Faz F (Colab) — Saldırılar ve canlı koşu

**durum: yapılmadı**

`attacks/`: random_weights, scaled_poison (10x/50x/100x),
conditional_poison (mapping'in DR-0 ve DR-4 gömme satırlarını
takasla). Hepsi ağırlık seviyesinde, eğitim gerektirmez. Bu adım tam G
ağırlıkları üzerinde çalışır (FID için tam generator lazım), Colab'da
koşar.

Her senaryo için: korumasız FedAvg -> FID/KID, ZK kapılı FedAvg ->
FID/KID, ispat saldırıyı yakaladı mı.

`eval/metrics.py`: fid50k_full, KID (sınıf başına), LPIPS (sınıf
başına). Mevcut ölçüm ayarlarıyla birebir aynı olmalı, yeni yol icat
etme — sonuçlar önceki deneylerle karşılaştırılabilir kalmalı.

Ayrıca 3 round canlı entegre koşu.

**Kabul:** saldırı matrisi + canlı koşu logu.
