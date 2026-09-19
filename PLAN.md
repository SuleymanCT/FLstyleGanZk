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

**durum: tamamlandı.** C0/C1/C2/C3 Colab'da tam olarak koştu ve
doğrulandı (bkz. aşağıdaki notlar + `docs/phase_c_report.md`, kapanış
raporu). Operasyonel devre parametreleri `configs/circuit.yaml`'a
yazıldı: `challenge_size_k=1`, `embed_mode=matmul`, `scale=8` —
gerekçeler C3 notunda ve dosyanın kendi yorumlarında. (C4 —
kuantizasyon etkisinin DR sınıfı başına ayrıntılı analizi — bu kapsamın
dışında bırakıldı, ayrı bir iş olarak ele alınabilir; Faz C'nin ana
kabul kriteri zaten C3'te karşılandı.) **Sıradaki faz: Faz D
(kontratlar ve orkestratör, k=1 protokolüyle).**

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

  **Colab sonucu (1. tur):** parametre 669.248→350.848 (%47,6 azalma),
  budanmış vs tam çıktı `max_abs_diff=0.0`, ONNX 35 düğüm, `onnxruntime`
  vs torch/referans ~2e-6. Grafikte `ArgMax`+`Gather`+`Cast` VARDI —
  ezkl için riskli.

  **Düzeltme (2. tur):** `circuits/rebuild_mapping.py`'ye `embed_mode`
  seçeneği eklendi — `"gather"` (orijinal: `embed(c.argmax(1))`) ve
  `"matmul"` (`c @ embed.weight` — `c` one-hot olduğundan matematiksel
  olarak ÖZDEŞ, `compare_embed_modes` ile 1e-6 toleransla doğrulanıyor;
  yerel sentetik testte gerçek fark 0.0 çıktı). `circuits/export_mapping.py`
  artık HER k için İKİ varyant ihraç ediyor: `mapping_k{k}_gather.onnx`
  ve `mapping_k{k}_matmul.onnx`; ikisinin op dağılımı da
  `{onnx_dir}/onnx_variants.json`'a yazılıyor (C3 benchmark tablosu
  için ham veri). Moddan bağımsız eski isim (`mapping_k{k}.onnx`)
  artık `matmul` varyantının kopyası (ArgMax/Gather İÇERMEZ) — bu
  yerelde `ArgMax`/`Gather`'ın `matmul` modunda GERÇEKTEN kaybolduğu
  doğrulanarak test edildi (`tests/test_export_mapping.py`). Gerçek
  StyleGAN-XL ağırlıklarıyla iki varyantın ezkl'e karşı nihai
  davranışı (hangisi derleniyor, kısıt sayısı/süre farkı) C3'te
  ölçülecek — kullanıcının notu: "makalede somut bir devre optimizasyonu
  bulgusu olacak", her iki varyant BİLEREK tutuluyor, ikisi de C3'e
  girecek.

  **Colab sonucu (2. tur, doğrulandı):** iki varyant da başarıyla ihraç
  edildi. `gather`: 35 düğüm, `{ArgMax:1, Gather:1, Cast:1, MatMul:3, ...}`.
  `matmul`: 35 düğüm, `{MatMul:4, Cast:2, ...}` — `ArgMax`/`Gather`
  GERÇEKTEN yok. İki mod arası çıktı farkı `0.0`, C0 referansına karşı
  ~2e-6. Budanmış parametre sayısı: 350.848 (1. turdakiyle aynı).

- **C3** `scripts/bench_circuit.py` — kod yazıldı, Colab'da HENÜZ
  koşulmadı. Izgara: k ∈ {1,4,8} × embed_mode ∈ {matmul,gather} ×
  `input_scale`/`param_scale` ∈ {8,11,13} (26. maddedeki "bit" fikri,
  ezkl==23.0.5'in gerçek API'sindeki karşılığı olan `input_scale`/
  `param_scale`'e göre güncellendi). Sıralama BİLEREK küçükten büyüğe:
  ilk kombinasyon (k=1, matmul, scale=8) — `--only-first` bayrağıyla
  tek başına denenebilir, büyük kombinasyonlara ancak bu geçerse geçilir.

  Her kombinasyon KENDİ alt sürecinde çalışır (üç gerekçe: (1) tepe RAM
  `resource.getrusage(RUSAGE_SELF).ru_maxrss` ile kombinasyon başına
  temiz ölçülebiliyor — aynı süreçte ardışık koşulsa önceki
  kombinasyonlardan ayrıştırılamazdı; (2) ezkl'nin Rust tarafının
  bastığı loglar [decomposition uyarıları, `max_abs_error`] Python
  `sys.stdout` yönlendirmesiyle değil, gerçek alt süreç stdout/stderr
  fd'siyle güvenilir yakalanıyor; (3) bir kombinasyon çökerse/segfault
  verirse sadece o alt süreç ölür, ebeveyn diğerlerine devam eder).
  Kombinasyon başına varsayılan timeout 1800s (konfigüre edilebilir),
  aşılırsa POSIX'te süreç GRUBU (`os.killpg`) öldürülüyor — anvil gibi
  torun süreçler de öksüz kalmıyor. Her kombinasyondan hemen sonra
  `bench_results.json` diske yazılıyor (atomic tmp+`os.replace`) —
  Colab kopsa bile zaten sonuçlanmış kombinasyonlar `--force` verilmeden
  tekrar koşulmuyor (resume).

  Ölçülen alanlar: `gen_settings`+`calibrate_settings` süresi (ayrı ayrı
  de saklanıyor), `compile_circuit` süresi + best-effort devre
  istatistikleri (`settings.json`'dan `logrows`/`num_rows`/
  `num_required_lookups` vb. — şema sürüme göre değişebileceğinden
  hiçbiri zorunlu değil), `setup` süresi + pk/vk boyutu, `gen_witness`/
  `prove`/`verify` (offchain) süreleri + ispat boyutu, decomposition
  uyarı SAYISI + `max_abs_error` (alt sürecin TAM stdout'undan regex ile
  ayrıştırılıyor), Solidity verifier üretimi+derlemesi (`chain.solc.compile_with_fallback_strategies`
  yeniden kullanıldı, `viaIR=False/runs=200` ilk sırada) + deployed
  bytecode boyutu + EIP-170 aşımı, `verify_gas` (anvil'de deploy+
  `verifyProof`, `circuits.toy_pipeline`'ın Faz B'de kanıtlanmış
  fonksiyonları — `generate_solidity_verifier`, `deploy_and_verify_onchain`,
  `deploy_and_verify_via_ezkl_native` fallback'ı — DOĞRUDAN yeniden
  kullanıldı, DRY).

  **Bilinçli varsayım (Colab'da doğrulanacak):** `calibrate_settings`
  scale'i otomatik seçtiğinden, ızgaranın `scale` ekseninin GERÇEKTEN
  uygulandığından emin olmak için `calibrate_settings` sonrası
  `settings.json` okunup `run_args.input_scale`/`param_scale` istenen
  değere ZORLANIYOR (`force_settings_scale`) — `run_args` beklenen
  şekilde değilse net UYARI basılıp calibrate'in seçimi kullanılıyor.
  Çoklu-girdi (`z`,`c`) `input.json` şeması `{"input_data": [flat_z,
  flat_c]}` olarak varsayıldı — ilk Colab koşumu bunu doğrulayacak.

  Saf yardımcılar (`build_grid`, `combo_key`, `count_decomposition_warnings`,
  `extract_max_abs_error`, `extract_circuit_stats`, `force_settings_scale`,
  `build_multi_input_json`, `load_bench_results`/`save_bench_results`,
  `render_markdown_table`) yerelde `tests/test_bench_circuit.py` ile
  GERÇEKTEN test edildi (18 test) — modül kendi top-level'ında `import ezkl`
  YAPMIYOR (sadece `run_ezkl_pipeline`/`run_prove_and_verify`/`run_worker`
  içinde, çağrıldıklarında), bu yüzden `--help` de dahil yerelde hatasız
  çalışıyor. `run_worker`/zincir adımları BİLİNÇLİ TEST SINIRI içinde
  (ezkl+anvil+solc+web3 gerektirir, sadece Colab'da doğrulanabilir).

  Çıktı: `{zk_root}/bench/bench_results.json` + konsol tablosu +
  `docs/phase_c_bench.md` (markdown tablo, aynı `render_markdown_table`
  fonksiyonuyla üretiliyor — DRY).

  **Colab sonucu (1. deneme, `--only-first`):** ilk kombinasyon (k=1,
  matmul, scale=8) `calibrate_settings`'te düştü: `RuntimeError: Failed
  to calibrate settings: [Uncategorized] calibration failed, could not
  find any suitable parameters given the calibration dataset` (tepe RAM
  945 MB, decomposition uyarısı 0). İki olası sebep var: (a) scale=8 dar,
  (b) çoklu-girdi `input.json` şeması yanlış. **`scripts/diagnose_calibration.py`**
  yazıldı — önce (b)'yi ele alıyor: `ezkl==23.0.5` (`v23.0.5` etiketi,
  WebFetch ile GERÇEKTEN çekildi, versiyon kayması riski yok) kaynağından
  doğrulandı: `src/graph/input.rs`'teki `DataSource = Vec<Vec<FileSourceInner>>`
  bizim `{"input_data": [flat_z, flat_c]}` şemamızı destekliyor GİBİ
  görünüyor; ayrıca `ezkl.pyi`'de `calibrate_settings`'in `scales: Optional[Sequence[int]]`
  ve `max_logrows: Optional[int]` parametrelerinin GERÇEKTEN var olduğu
  görüldü — `bench_circuit.py`'nin şu anki `force_settings_scale`
  (calibrate SONRASI settings.json'u elle yamama) yaklaşımı yerine
  doğrudan `scales=[scale]` verilebilir olabilir. Script hem bunu hem
  eski yöntemi (`run_args.input_scale`) ampirik olarak karşılaştırıyor
  (scale×target×yöntem ızgarası + `max_logrows` taraması), hem şema/
  şekil/değer aralığını gerçek verilerle gözle doğruluyor (ONNX grafiğinin
  girdi şekli vs input.json uzunluğu çapraz kontrolü, z/c/w gerçek min/
  max/mean), hem de kalibrasyonu TAMAMEN atlayıp sabit scale ile
  compile→setup→witness→prove'u dener (kalibrasyon şart mı sorusu).
  Sonuçlar `{zk_root}/bench/diagnose/diagnose_report.json`'a yazılıyor.
  Saf yardımcılar (`describe_tensor_stats`, `describe_input_json_shape`,
  `read_onnx_input_shapes`, `cross_check_input_shapes`, `parse_max_logrows_values`,
  tablo yazıcılar) yerelde 11 testle GERÇEKTEN doğrulandı; ezkl'e dokunan
  kısımlar (`dump_full_settings`/`attempt_calibration`/`attempt_skip_calibration`)
  BİLİNÇLİ TEST SINIRI içinde, sadece Colab'da koşacak. `bench_circuit.py`'nin
  kendisi bu turda DEĞİŞTİRİLMEDİ — teşhis sonucu gelene kadar bilerek
  ertelendi.

  **Teşhis sonucu (Colab, ÇÖZÜLDÜ) — bkz. `docs/phase_c_calibration.md`
  (üç bulgu detaylı):**
  1. Şema DOĞRUYMUŞ (b elendi); doğru yöntem `calibrate_settings(...,
     scales=[scale])` — bu Colab'da GERÇEKTEN ÇALIŞTI (scale=8,
     `max_abs_error=0.133`, `logrows=19`). Eski `force_settings_scale`
     yöntemi (calibrate SONRASI settings.json'u elle yamama) hiçbir
     ölçekte tutmamış — `bench_circuit.py`'den KALDIRILDI, yerine
     `read_realized_scale` (sadece okur, zorlamaz) geldi.
  2. Beklenenin TERSİ: scale=8 çalışıyor, scale=11/13 `calibrate_settings`'te
     düşüyor ("[halo2] General synthesis error" / "significant bit
     truncation"), scale=16 GERÇEK BİR RUST PANİĞİ fırlatıyor
     (`pyo3_runtime.PanicException`). Muhtemel sebep: `mapping.fc1`'in
     512 terimlik iç çarpımları yüksek ölçekte decomposition tabanını
     aşıyor — makale için somut bir "devre boyutu büyüdükçe kullanılabilir
     scale aralığı daralıyor" bulgusu. Izgara `{8,11,13}`'ten `{6,7,8,9,10}`'a
     değiştirildi (çalışan sınırın etrafında ince tarama).
  3. `PanicException` `BaseException`'dan türüyor, `except Exception`
     yakalamıyordu — hem `bench_circuit.py: run_worker` hem
     `diagnose_calibration.py`'nin ilgili fonksiyonları artık
     `except (KeyboardInterrupt, SystemExit): raise` + `except
     BaseException` deseniyle bunu yakalayıp `status="panic"` (sıradan
     `"failed"`'den ayrı) işaretliyor, koşu düşmüyor
     (`classify_exception_status`, iki script'te de ortak kullanılıyor).
  4. `bench_circuit.py` artık her kombinasyon için `max_abs_error`'u
     referans `w`'nin gerçek büyüklüğüne oranlayıp `max_abs_error_relative_pct`
     (scale=8'de ~%12,5) olarak da raporluyor — scale/fidelity ödünleşim
     tablosu makale için hazır hale geldi.

  **Disk kotası sorunu (Colab, ÇÖZÜLDÜ):** ilk gerçek kombinasyon geçti
  ama `pk.key` **2,5 GB** çıktı — 30 kombinasyonluk tam ızgarada Drive
  kotasını doldururdu. İki önlem: (1) çalışma kökü artık varsayılan
  olarak **`/content`** altında (Colab'ın yerel VM diski, Drive DEĞİL,
  `default_work_root()`, `--work-root` ile değiştirilebilir) — sadece
  küçük sonuç dosyaları (`bench/results/{key}.json`, `.log.txt`,
  `bench_results.json`) `--bench-dir`'e (Drive) yazılıyor. (2) Worker,
  ölçümleri kaydettikten SONRA `cleanup_work_dir` ile büyük ara
  dosyaları (`pk.key`, `network.compiled`, `witness.json`) siliyor —
  boyutları zaten `result.json`'da. `settings.json`/`proof.json`/
  `Verifier.sol`/`vk.key`/`input.json` korunuyor. `--keep-artifacts`
  ile temizlik tamamen kapatılabiliyor. Her kombinasyon sonrası
  temizlik öncesi/sonrası disk kullanımı loglanıp `result.json`'a
  (`disk_usage_before/after/freed_cleanup_bytes`) yazılıyor —
  `render_markdown_table`'a `disk_oncesi(MB)`/`disk_sonrasi(MB)`
  sütunları eklendi. NOT: SRS dosyası bu temizlikte YOK — `srs_path`
  hiç verilmediğinden (`None`) ezkl kendi varsayılan önbelleğini
  kullanıyor, work_dir'e hiç yazılmıyor, zaten Drive'a gitmiyor.

  **TAM IZGARA SONUCU (Colab, 30/30 kombinasyon koştu) — bkz.
  `docs/phase_c_report.md` (tam analiz):**
  - **Ölçek tavanı = 8** (her k/embed_mode'da tutarlı): scale 6/7/8 hep
    geçiyor, 9/10 hep kalibrasyonda düşüyor. `decomposition_uyari` HER
    kombinasyonda 0 — gerçek devre sert bir eşikte duruyor, oyuncak
    modelin "yavaşça bozulma" davranışını göstermiyor.
  - **matmul, gather'dan ~2× ucuz** (k=1, scale=8): prove 37.1s/75.5s,
    setup 36.0s/69.6s, pk 2.52GB/5.61GB, tepe RAM 7.3GB/17.3GB, verifier
    bytecode 14.929B/19.735B, verify_gas 1.174.812/1.350.486 — somut
    devre-optimizasyonu bulgusu (C2'nin öngördüğü gibi).
  - **EIP-170: sadece k=1 geçiyor** (14.929/19.735 byte); k=4/k=8
    AŞIYOR (26.757/26.758, 32.144/32.142 byte) — zincir üstü doğrulama
    sadece k=1 ile mümkün. **Protokol sonucu: Faz D her challenge'da TEK
    örnek ispatlamalı** (`configs/circuit.yaml: challenge_size_k=1`).
  - **3 hata düzeltildi (1. tur):** (1) `ezkl.deploy_evm` private key
    `0x` önekiyle çağrılıyordu, `deploy_evm` önekSİZ 64-hex bekliyor —
    `chain/anvil.py: normalize_private_key_hex` eklendi. (2)
    `max_abs_error`/`max_abs_error_%` boştu — ezkl'nin GERÇEK raporu
    basit `k=v` değil çok sütunlu pipe'lı markdown tablo;
    `parse_markdown_table_column` (yeni) sütun-bazlı okuyor, hâlâ
    ayrıştırılamazsa `"yakalanamadi"` yazıyor (boş bırakmıyor). (3)
    `gerceklesen_scale` boştu — basit bir kopyalama hatası
    (`run_worker`, `setup_result["realized_scale"]`'i `result`'a hiç
    kopyalamıyordu), düzeltildi. `--only` bayrağı eklendi (belirli
    kombinasyonları yeniden koşmak için, 30'unu baştan koşmadan).

  **Hedefli yeniden koşum (2. tur) — gerceklesen_scale/max_abs_error
  DOĞRULANDI, YENİ bir deploy hatası çıktı:** `max_abs_error` (matmul,
  scale=8): k=1→0.1331 (%3,21), k=4→0.137 (%3,205), k=8→0.1658
  (%3,879) — bkz. `configs/circuit.yaml` ve `docs/phase_c_report.md`
  "Fidelity" bölümü (ispatın geçerliliğini ETKİLEMİYOR, kuantize
  hesabın kendisi doğrulanıyor). private-key düzeltmesi işe yaradı ama
  k=4/k=8'de YENİ hata çıktı: `[eth] failed to parse url .../Verifier.sol`
  — `ezkl.deploy_evm`'in GERÇEK pozisyonel argüman sırası `ezkl.pyi`'nin
  belgelediğinden FARKLIYMIŞ (`sol_code_path` `rpc_url` slotuna
  düşüyordu). **Düzeltme:** çağrı artık pozisyonel değil, `ezkl.pyi`'den
  alınan parametre ADLARIYLA (kwarg). Ayrıca MİMARİ düzeltme: EIP-170
  aşımı artık solc "tamamen başarısız"dan AYRI ele alınıyor — bytecode
  sınırı aşıyorsa deploy HİÇ DENENMİYOR, `status="eip170_exceeded"`
  (`"failed"`den ayrı) işaretleniyor (eskiden EIP-170'in DOĞAL deploy
  reddi yanlışlıkla "solc başarısız" sanılıp ezkl'nin native
  `deploy_evm`'ine düşürülüyordu — asıl kök sebep buydu).

  **DOĞRULANDI (3. tur, `--only k4_matmul_scale8,k8_matmul_scale8 --force`):**
  temiz sonuç — k4 ve k8, `status="eip170_exceeded"`, `error_summary=None`.
  Ne private-key ne url-parse hatası; deploy hiç denenmedi. **Faz C3
  KAPANDI.** Operasyonel yapılandırmanın (k=1, matmul, scale=8) nihai
  ölçümü: setup 37.3s, prove 37.4s, tepe RAM 7.2GB, proof 96KB,
  verifier bytecode 14.929B, `verify_gas=1.174.788` (zincir üstü
  doğrulandı), bağıl hata %3,21 — tam tablo `docs/phase_c_report.md`'nin
  başında. **NOT:** k4/k8'in `gather` varyantları yeniden koşulmadı,
  `bench_results.json`'da hâlâ eski private-key hatalı `"failed"`
  durumunda — ama zaten `matmul` ile aynı k'de bytecode boyutu neredeyse
  özdeş ve EIP-170'i AŞIYOR, yeniden koşulsa da sonuç değişmez
  (`eip170_exceeded` olurdu); Faz C'nin kapanışını etkilemiyor,
  istenirse ayrıca doğrulanabilir.

- **C4** Kuantizasyon etkisi: w_q ve fp32 w arasında kosinüs benzerliği
  ve L2 farkı, DR sınıfı başına ayrı. **Faz C'nin kapsamı dışında
  bırakıldı** — ayrı bir iş olarak ele alınabilir.

**Kabul:** en az bir (k, bit) kombinasyonu 30 dk altında ispat
üretiyor. **KARŞILANDI** — k=1, matmul, scale=8: setup 36.0s + prove
37.1s ≈ 73s toplam, 30 dk'nın çok altında.

**Faz C2'den not (C3 için girdi — ÇÖZÜLDÜ, iki varyant hazır):** İhraç
edilen ONNX grafiğinde `ArgMax`+`Gather`+`Cast` vardı (`c.argmax(dim=1)`
ile sınıf indeksini bulup `embed`'den satır çekmekten geliyor). Çözüm
uygulandı: `embed_mode="matmul"` (`c @ embed.weight`, `c` one-hot
olduğundan matematiksel olarak özdeş) ile aynı devrenin `ArgMax`/`Gather`'sız
bir varyantı üretiliyor. **C3'ün ilk adımı artık ikisini de ayrı ayrı
`ezkl.gen_settings`/`compile_circuit`'e vermek olmalı** —
`mapping_k{k}_gather.onnx` derlenemezse (ya da derlenip aşırı yavaşsa),
`mapping_k{k}_matmul.onnx` fallback değil ANA yol olur. İkisi de
BİLEREK tutuluyor (kullanıcı: "makalede somut bir devre optimizasyonu
bulgusu olacak") — C3'ün (k, bit) benchmark ızgarasına `embed_mode`
(gather/matmul) da bir boyut olarak eklenmeli, sadece k×bit değil.
**(KOD YAZILDI:** `scripts/bench_circuit.py`'nin ızgarası
k×embed_mode×scale — `embed_mode` sıralamada `matmul` önce geliyor,
"en küçükten başla" ilkesiyle tutarlı.**)**

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
**(KOD YAZILDI:** `scripts/bench_circuit.py: count_decomposition_warnings`/
`extract_max_abs_error` alt sürecin TAM stdout'unu regex ile ayrıştırıp
her kombinasyon için ayrı sütun olarak `bench_results.json`'a yazıyor;
"bit_width" yerine ezkl==23.0.5'in gerçek API'si `input_scale`/`param_scale`
kullanılıyor, ızgara {8,11,13}.**)**

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
**(KOD YAZILDI:** `scripts/bench_circuit.py` her kombinasyonda
`compile_verifier_solidity` sonucunu `deployed_bytecode_size`/
`exceeds_eip170` olarak kaydediyor; ayrıca EIP-170'i aşan bytecode'un
gerçek deploy denemesi zaten EVM tarafından reddedilip (`receipt.status==0`)
`Web3Client.deploy_bytecode`'un fırlattığı hata üzerinden kombinasyon
doğal olarak "failed" işaretleniyor — özel bir erken-çıkış eklenmedi,
gerçek zincir davranışı gözlemleniyor.**)**

## Faz D (yerel) — Kontratlar ve orkestratör

**durum: tamamlandı.** `tests/test_contracts.py`'nin altı senaryosu da
(6/6) Colab'da, GERÇEK ezkl ispatlarıyla ve GERÇEK bir Halo2Verifier'a
karşı doğrulandı — bkz. `docs/phase_d_report.md` (mimari, ölçülen
değerler, dört veri-biçimi tuzağının tam analizi). Protokol k=1 ile
çalışır (Faz C3'ün EIP-170 bulgusu — bkz. `docs/phase_c_report.md`).

**ÖNEMLİ TASARIM SAPMASI (kullanıcının orijinal spesifikasyonundan,
gerekçeli):** `RoundManager.sol` TEK bir sabit `verifier` adresi
TUTMUYOR — `submitProof(roundId, verifierAddress, proof, publicInputs)`
verifier adresini PARAMETRE olarak alıyor. Sebep: Faz B/C boyunca ezkl
`param_visibility="fixed"` kullanıldı — ağırlıklar devrenin SABİT
sütunlarına (vk/verifier bytecode'unun İÇİNE) gömülü. Federe öğrenmede
her round/site'ın ağırlıkları FARKLI olduğundan HER (round,site)
ispatının KENDİ verifier'ı olması GEREKİYOR — tek bir sabit adres round
2'den itibaren YANLIŞ olurdu. `orchestrator/round_runner.py` bu yüzden
her gerekli ispat için TAZE bir Verifier deploy edip adresini
`submitProof`'a veriyor (`Submission.verifierUsed`'da denetlenebilir
tutuluyor). Bunun somut bir maliyet sonucu var: HER gerekli ispat artık
ezkl setup+prove (~75s, Faz C3) + solc derleme (~20s, Faz B) + verifier
deploy gas'ı (~2,96M, Faz B) + submitProof gas'ı (~1,17M, Faz C3)
gerektiriyor — `orchestrator/schedule.py`'nin `random_ratio=0.3`
örneklemesinin (her round her site yerine) neden ekonomik olarak
GEREKLİ olduğunu somutlaştırıyor.

- **`contracts/RoundManager.sol`**: `registerSite` (onlyOwner),
  `startRound` (challengeSeed = `keccak256(blockhash(block.number-1), roundId)`),
  `submitUpdate` (kayıtlı site + açık round + tekil gönderim kontrolü),
  `submitProof` (yukarıdaki tasarım notuyla; `IHalo2Verifier.verifyProof`'u
  `try/catch` ile çağırıyor — ezkl'nin ürettiği GERÇEK Verifier'ın imzası
  kesin bilinmediğinden, `try/catch` HER İKİ olası davranışa (revert EDER
  ya da `false` DÖNER) karşı güvenli: en kötü ihtimalle ispat "geçersiz"
  sayılır, asla yanlış pozitif üretmez — imza gerçekten uyuşmuyorsa
  Colab'daki `tests/test_contracts.py` bunu AÇIKÇA ortaya çıkaracak),
  `finalizeRound` (sadece `proofVerified=true` VEYA ispat hiç
  istenmemiş site'lar dahil edilebilir), itibar +bonus/-penalty +
  eşik-altı `SiteExcluded` event'i, `getChallengeSeed`/`isSiteEligible`/
  `getRoundInfo` + ek `getSubmission` (spesifikasyonda yoktu ama
  `proofVerified`'ı zincirden geri okumak için gerekli). Her durum
  değişikliğinde event. Derleme: `chain.solc.compile_with_fallback_strategies`
  (Faz B'nin viaIR=False/runs=200/solc=0.8.20 ilk sırada).

- **`chain/client.py`**: `RoundManagerClient` (yeni, `Web3Client`'tan
  türer) — `RoundManager.sol`'un her fonksiyonu için bir metod, her
  state-değiştiren çağrı `(sonuç, gasUsed)` döner. Private key'ler
  `normalize_private_key_hex` ile normalize edilip geri `0x` eklenerek
  veriliyor (Faz C3'ün private-key tuzağına burada da düşmemek için —
  web3.py aslında ikisini de kabul ediyor, ama erken doğrulama için
  bilerek kullanıldı). `Web3Client.deploy_contract` (yeni) constructor
  argümanlarını web3.py'nin KENDİ ABI kodlamasıyla (elle DEĞİL) ekliyor.

- **`orchestrator/challenge.py`**: `build_challenge_z_c` —
  `fl.reference_utils.build_z_c`'yi (Faz C0) DOĞRUDAN yeniden kullanır,
  tohum artık zincirden gelen `bytes32`'nin türevi. k=1 varsayılan
  (Faz C3). 8 test, saf, yerelde gerçekten koşuyor.

- **`orchestrator/schedule.py`**: `must_prove` üç kuralı OR'lar (sabit
  round → zorunlu, itibar eşik-altı → zorunlu, aksi halde
  `challengeSeed`+round+site'den deterministik-sha256 ile `[0,1)` bir
  değer ürettirip `random_ratio` ile karşılaştırır). `mode="full"`
  (her round her site — makalenin karşılaştırma tabanı) vs
  `mode="sampled"` (üretim modu). `configs/schedule.yaml` gerçek
  operasyonel değerlerle dolduruldu: `fixed_rounds=[0,7,14]`,
  `random_ratio=0.3`, `reputation_threshold=50`, `reputation_initial=100`,
  `reputation_penalty=20`, `reputation_bonus=1` — her biri dosyanın
  içinde gerekçeli. 13 test, saf, yerelde gerçekten koşuyor.

- **`orchestrator/aggregate.py`**: `fl.fedavg_utils.RunningAverage`'i
  DOĞRUDAN kullanır (yeniden YAZMAZ) — bu, Faz A'nın `audit_fedavg.py`'sinin
  doğruladığı fp32 aritmetikle BİREBİR aynı sonucu TANIM GEREĞİ garanti
  eder. `gate_site_update`: norm (`tau_norm_threshold=3000.0`, Faz A'nın
  p99 ölçümü), NaN/Inf, zincir-onayı kontrolleri — KUANTİZASYON YAPMAZ
  (CLAUDE.md madde 5). `test_aggregate_round_matches_running_average_when_nothing_excluded`
  bu garantiyi doğrudan `RunningAverage`'e karşı kanıtlıyor. 13 test,
  saf, yerelde gerçekten koşuyor.

- **`storage/ipfs.py`**: yerel kubo node'u için ince `add`/`get`
  sarmalayıcısı, node yoksa net hata (uydurma CID YOK). `scripts/setup_colab.sh`'a
  kubo (v0.29.0) kurulumu eklendi. Saf `is_valid_cid` yerelde test
  edildi.

- **`fl/round_replay.py`**: `round_runner.py`'nin "yerel eğitim" adımının
  ince sarmalayıcısı — **GERÇEK EĞİTİM YAPMAZ** (CLAUDE.md madde 2/3:
  base model/15 round yeniden eğitilmez, eğitim döngüsüne dokunulmaz).
  Faz A'da ZATEN üretilmiş site snapshot'ını `raw_root`'tan (salt okunur)
  yükler — `round_runner.py` canlı bir eğitim turu BAŞLATMIYOR, mevcut
  bir round'u protokol açısından REPLAY ediyor (canlı entegre koşu ayrı,
  Faz F'in kapsamı).

- **`scripts/deploy_contracts.py`**: SADECE `RoundManager`'ı deploy eder
  (yukarıdaki tasarım sapması gereği genesis'te bir Verifier YOK — her
  gerekli ispat kendi Verifier'ını `round_runner.py` üzerinden alır).
  anvil bu script tarafından yönetilmiyor (bilerek — deploy edilen
  kontrat script bittikten SONRA da durmalı), `--rpc-url`/`--private-key`
  ZORUNLU.

- **`orchestrator/round_runner.py`**: 9 adımlık akış (bkz. modül
  docstring'i) — `progress.json` ile devam edilebilir (atomic
  tmp+`os.replace`, `extract_shards.py`'nin deseniyle tutarlı). Her
  gerekli ispat için `scripts.bench_circuit.run_ezkl_pipeline`/
  `run_prove_and_verify` + `circuits.toy_pipeline.generate_solidity_verifier`/
  `compile_verifier_solidity` DOĞRUDAN yeniden kullanılıyor (DRY, hiçbir
  ezkl/solc çağrısı yeniden yazılmadı). `proof.json`'un İÇERİĞİ
  (`{"proof": "0x...", "instances"/"public_inputs": [...]}` varsayıldı)
  Faz D'de İLK KEZ ayrıştırılıyor — Faz B/C boyunca sadece dosya varlığı
  kontrol ediliyordu, içeriği hiç okunmamıştı; bu VARSAYIM Colab'da
  doğrulanacak.

- **`tests/test_contracts.py`**: mutlu yol, geçersiz ispat reddi +
  itibar düşüşü, eşik-altı dışlama, çift submit reddi, kayıtsız site
  reddi, doğrulanmamış site'nin finalizeRound'a alınamaması. GERÇEK
  ezkl ispatlarıyla (Drive/pkl GEREKMEYEN küçük sentetik bir devreyle —
  testi Colab'a özgü olmaktan çıkarıyor, ama yine de ezkl'nin GERÇEK
  setup/prove/verify zincirinden geçiyor — "geçersiz ispat" testi
  GERÇEK bir proof'u bilerek bozuyor, sıfırdan uydurma bytes DEĞİL).
  Pahalı ezkl+solc kurulumu modül başına BİR KEZ (`scope="module"`
  fixture), testler arasında paylaşılıyor.

**BİLİNÇLİ TEST SINIRI (güncellendi):** `contracts/RoundManager.sol` +
`chain/client.py: RoundManagerClient`'in kontrat çağrı yolları artık
Colab'da GERÇEKTEN doğrulandı (`tests/test_contracts.py`, 6/6). Hâlâ
yerelde çalıştırılamayan/test edilemeyen (ve HENÜZ Colab'da da gerçek
bir uçtan uca koşumla doğrulanmamış) kısımlar: `orchestrator/round_runner.py`'nin
TAM 9 adımlık akışı (gerçek `raw_root` pkl replay'i + gerçek IPFS
upload/download ile), `scripts/deploy_contracts.py`, `storage/ipfs.py`'nin
gerçek bir kubo node'una karşı çalışması — bunlar `tests/test_contracts.py`'nin
kapsamadığı, sentetik/izole test yerine GERÇEK bir round koşumu
gerektiren parçalar (bkz. Faz E). Saf/dosya-tabanlı yardımcılar (`challenge.py`,
`schedule.py`, `aggregate.py`, `round_runner.py`'nin `load_progress`/
`save_progress`/`compute_weight_commitment`/`load_circuit_config`'i,
`storage/ipfs.py: is_valid_cid`) 40+ testle yerelde GERÇEKTEN doğrulandı.
`web3` bu turda yerelde de kuruldu (`onnx`/`onnxruntime` emsaliyle —
Colab'a özgü değil, sade bir Python paketi) — `chain/client.py: _build_tx`
artık yerelde de GERÇEKTEN test ediliyor (222 passed, 2 skipped toplam).

**1. Colab koşumu — TEK hata, düzeltildi:** ezkl ispatı üretildi,
Verifier.sol (14.921 byte) + RoundManager.sol (6.867 byte) derlendi,
anvil ayağa kalktı, `deploy_bytecode` ile Verifier deploy edildi — TEK
hata `RoundManagerClient.deploy_contract`'ta: `TypeError: Unknown kwargs:
['gasPrice']` (işlem sözlüğünde hem EIP-1559 hem legacy gas alanı
bir aradaydı). Kök sebep: `factory.constructor(...).build_transaction({})`
anvil'in EIP-1559 desteğini görüp `maxFeePerGas`/`maxPriorityFeePerGas`'ı
KENDİSİ ekliyordu, `_send_and_wait`/`RoundManagerClient._send` ise bunun
ÜSTÜNE KOŞULSUZ `gasPrice` de ekliyordu. **Düzeltme:** `chain/client.py: _build_tx`
— TEK ortak yol, `base_tx`'te hangi tip zaten varsa ona sadık kalır,
hiçbiri yoksa EIP-1559 ekler, ikisi birden varsa `ValueError` ile durur
(sessizce "düzeltmez"). `_send_and_wait`/`deploy_contract`/`RoundManagerClient._send`
artık HEPSİ bu tek yoldan geçiyor. `tests/test_client.py` (6 test,
sentetik `w3` nesnesiyle) bu iki gas biçiminin ASLA bir arada
üretilmediğini yerelde GERÇEKTEN kanıtlıyor.

**2. Colab koşumu — deploy_contract düzeldi, kontrat mantığı GERÇEKTEN
çalıştığı doğrulandı** (gerçek revert mesajları görüldü: "RoundManager:
bu round'a zaten gonderim yapildi", "RoundManager: kayitli site degil").
İki kalan sorun düzeltildi:

- **`public_inputs` tipi:** `proof.json`'daki `instances` iç içe liste,
  elemanlar hex string VE **LITTLE-ENDIAN** (`int(x,16)` ile okunursa
  devasa/yanlış bir sayı çıkıyordu — kontrat `uint256[]` beklerken
  `"Argument 4 value [...] is not compatible with type uint256[]"` hatası
  çıktı). **Düzeltme:** `circuits/ezkl_utils.py: parse_public_inputs`
  (yeni) — iç içe listeyi düzleştirir, hem little hem big-endian
  yorumunu hesaplayıp HANGİSİ BN254 skalar alanının (`BN254_SCALAR_FIELD_MODULUS`)
  İÇİNDE kalıyorsa onu kullanır (VARSAYMAZ, DOĞRULAR), ikisi de dışarıda
  kalırsa `ValueError`, zaten `int` olan elemanları olduğu gibi bırakır.
  `orchestrator/round_runner.py` ve `tests/test_contracts.py`'nin
  fragile inline ayrıştırması bununla değiştirildi. 14 test (little/big-
  endian ayırt etme, düz/iç-içe liste, zaten-int, karışık/eksik şema),
  saf, yerelde GERÇEKTEN doğrulandı.
- **Beklenen hata tipi:** testler `RuntimeError(match="İşlem başarısız")`
  bekliyordu ama web3 `ContractLogicError` fırlatıyordu — GAS TAHMİNİ
  aşamasında (`build_transaction()`'ın kendi iç `estimate_gas`
  simülasyonu revert'i doğrudan yakalıyor), işlem hiç zincire
  GÖNDERİLMEDEN. **Düzeltme:** `chain/client.py: RoundManagerClient._send`
  artık `fn_call.build_transaction(...)`'ı `try/except ContractLogicError`
  ile sarıyor, revert mesajını KORUYARAK (`str(e)`) aynı "İşlem başarısız"
  önekiyle `RuntimeError`'a çeviriyor — çağıran taraf (ve testler) TEK
  bir hata tipine bakabiliyor. `tests/test_contracts.py`'nin üç reddetme
  testi artık sadece hata TİPİNİ değil, kontratın GERÇEK revert
  metnini de (`"zaten gonderim yapildi"`, `"kayitli site degil"`,
  `"dogrulanmamis site finalizeRound'a dahil edilemez"`) doğruluyor.

**3. Colab koşumu — 5/6 test geçti, mutlu yolda TEK kalan hata:**
`public_inputs` düzeltmesi tuttu ("Argument 4 value [0, 2188824287...,
1, 1, 6] is valid"). Kalan hata: `"Argument 3 value [16, 209, 217, 7,
...] is not compatible with type bytes"` — `proof.json`'un `proof`
alanı da (`instances` gibi) beklenmeyen bir biçimdeydi: hex string
DEĞİL, bir INT LİSTESİ. `_shared_setup`'taki eski dönüşüm
(`isinstance(proof_json["proof"], str)`) sadece hex-string dalını
kapsıyordu, int listesini OLDUĞU GİBİ bırakıyordu (diğer 5 test
`bytes(corrupted)` ile ayrıca çevirdiği için sorunsuz geçmişti).
**Düzeltme:** `circuits/ezkl_utils.py: parse_proof_bytes` (yeni,
`parse_public_inputs`'un yanında) — int listesi (0-255 aralığı
doğrulanarak), hex string (`0x` önekli/öneksiz), zaten `bytes`/`bytearray`
olan durumların HEPSİNİ `bytes`'a çevirir, tanınmayan bir biçimde net
hata verir. `orchestrator/round_runner.py`/`tests/test_contracts.py`'nin
fragile inline dönüşümü bununla değiştirildi. `chain/client.py:
RoundManagerClient.submit_proof` artık `proof` parametresi `bytes`/`bytearray`
DEĞİLSE (sessizce çevirmeden) net bir `TypeError` veriyor — çağıran
tarafın `parse_proof_bytes` kullanması ZORUNLU. 9 yeni test (int listesi,
hex öneksiz/önekli, bytes, bytearray, eksik anahtar, tek-karakter hex,
aralık-dışı int, tanınmayan tip), saf, yerelde GERÇEKTEN doğrulandı
(241 passed, 2 skipped toplam).

**4. Colab koşumu — 6/6 test geçti, Faz D TAMAMLANDI:** `IHalo2Verifier.verifyProof(bytes,uint256[])
returns (bool)` imza VARSAYIMI, GERÇEK ezkl Verifier.sol'una karşı
TUTTU — `try/catch` hiçbir zaman "güvenli başarısızlık" dalına
düşmeden, gerçek `bool` dönüş değeri üzerinden çalıştı. Mutlu yol dahil
altı senaryonun TAMAMI (geçerli ispat kabulü, tek-bit-bozuk ispat
reddi + itibar düşüşü, eşik-altı dışlama, çift gönderim reddi, kayıtsız
site reddi, doğrulanmamış site'nin finalizeRound'a alınamaması) gerçek
revert mesajlarıyla doğrulandı. Ölçülen: RoundManager deployed bytecode
6.867 B, Verifier (test devresi) 14.921 B — gerçek k=1/matmul/scale=8
devresinin verifier'ına (14.929 B, Faz C3) neredeyse özdeş boyutta.
`contracts/RoundManager.sol`'daki `IHalo2Verifier` yorumu artık "varsayım"
değil "DOĞRULANDI" diyor; `try/catch` savunma katmanı olarak kalıyor.
Tam analiz + dört veri-biçimi tuzağının (gasPrice/EIP-1559 çakışması,
public_inputs little-endian iç içe liste, proof int listesi,
ContractLogicError'ın estimate_gas aşamasında oluşması) makale-hazır
anlatımı: `docs/phase_d_report.md`.

**Kabul: KARŞILANDI** — testler yeşil (6/6), gas rakamları raporlandı
(`docs/phase_d_report.md`).

## Faz E (yerel + Colab) — Replay

**durum: tamamlandı.**

`scripts/replay_proofs.py`: mevcut 15 round'un 60 shard'ı (`{shards_dir}/round_N/site_M.pt`,
Faz A çıktısı) üzerinde, HİÇ EĞİTİM YAPMADAN (CLAUDE.md madde 2/3),
ispat takvimini koşturur. İki mod: `"full"` (her round × her site = 60
ispat, karşılaştırma tabanı) ve `"staged"` (`orchestrator/schedule.py`'nin
takvimi: sabit turlar + rastgele seçim + itibar tetikli zorunlu ispat).
Devre parametreleri `configs/circuit.yaml` (k=1, matmul, scale=8 — Faz
C3), itibar/takvim parametreleri `configs/schedule.yaml`'dan (Faz D).

Mimari `scripts/bench_circuit.py`'nin (Faz C3) alt-süreç deseniyle
BİREBİR aynı gerekçelerle kuruldu (tepe RAM izolasyonu, ezkl'nin Rust
loglarının güvenilir yakalanması, çökme/panik izolasyonu) — GERÇEK ispat/
deploy/submit mantığı `orchestrator.round_runner.generate_and_submit_proof`'u
(Faz D'de `tests/test_contracts.py` ile doğrulanmış) DOĞRUDAN çağırır,
hiçbir ezkl/solc/chain adımı yeniden yazılmadı. Bu fonksiyon bu turda
GENİŞLETİLDİ: artık `pk_size_bytes`/`vk_size_bytes`/`proof_size_bytes`/
`circuit_stats`/`deployed_bytecode_size`/`solc_compile` süresi/`w_abs_max`
(bağıl hata hesabı için) de döndürüyor — hem `round_runner.py` hem
`replay_proofs.py` bundan yararlanıyor.

**anvil TEK bir süreçte, TÜM koşum boyunca canlı kalıyor** (her ispat
için yeniden başlatılmıyor) — `RoundManager` bir kez deploy ediliyor,
her round için GERÇEK bir `startRound` çağrısıyla GERÇEK bir
`challengeSeed` alınıyor. **"staged" modda hangi site'ların ispat
üreteceği ÖNCEDEN hesaplanamıyor** — `must_prove`'un rastgele bileşeni
GERÇEK `challengeSeed`'e bağlı olduğundan (o da GERÇEK `startRound`
çağrısı yapılmadan bilinemez), takvim kararı her round için GERÇEK
`challengeSeed` alındıktan HEMEN SONRA, çalışma zamanında veriliyor.

Dayanıklılık: `progress`/sonuç dosyası her ispattan sonra atomic
yazılıyor (`--force` verilmedikçe zaten tamamlanmış kombinasyonlar
atlanıyor), bir ispat çökerse/panikler se `status`/`error_summary` ile
işaretlenip diğerlerine devam ediliyor, `--keep-artifacts` yoksa büyük
ara dosyalar (pk, derlenmiş devre, witness) her ispattan sonra siliniyor,
`--only-first`/`--limit N` ile küçük ölçekte önce denenebiliyor.

Saf yardımcılar (`parse_int_range`, `replay_combo_key`, `compute_mode_totals`,
`render_mode_comparison_table`, `render_staged_distribution`,
`render_failure_summary`, `validate_args`) 24 testle yerelde GERÇEKTEN
doğrulandı (265 passed, 2 skipped toplam). ezkl/anvil/solc/gerçek shard
gerektiren kısımlar (`run_worker`, `main`'in chain dalı) BİLİNÇLİ TEST
SINIRI içinde — sadece Colab'da doğrulanabilir.

**1. Colab koşumu — altyapı çalıştı (RoundManager deploy gas=1.537.371,
startRound gas=122.029, challenge seed üretildi) ama işçi süreç
`status=failed`, `tepe_RAM_KB=None` döndü VE HATA MESAJI HİÇ BASILMADI.**
İki kök sebep: (1) `main()`'in per-kombinasyon özet satırı `error_summary`'yi
HİÇ YAZDIRMIYORDU (`bench_circuit.py`'nin aksine) — eklendi. (2) İşçinin
kendi `result.json`'u HİÇ YAZILMAMIŞTI, yani işçi `try/except/finally`
bloğunun (importlar ve `work_dir.mkdir()` dahil, eskiden bunlar `try`'ın
DIŞINDAYDI) dışında bir yerde tamamen çökmüştü — ebeveyn bunu "Worker
süreci sonuç dosyası yazmadan sonlandı" diye biliyordu ama BUNU DA hiç
yazdırmıyordu. **Düzeltme:** `run_worker` artık HİÇBİR kod yolu try/except
dışında bırakmıyor (importlar dahil), `finally` içindeki `cleanup_work_dir`/
`write_json_file` çağrıları da KENDİ try/except'leriyle korunuyor (biri
başarısız olursa sonuç KAYBOLMUYOR, en azından STDOUT'a basılıyor);
`result.json` hiç yoksa ebeveyn artık ayrı bir `"crashed"` durumu +
`returncode` + son 50 satır stdout kaydediyor; işçinin TAM çıktısı
başarısızlıkta KOŞULSUZ, başarıda `--verbose` ile gösteriliyor; başarısız
kombinasyon için komut TEKRAR (kopyala-yapıştır için) yazdırılıyor;
kapanış raporuna (`render_failure_summary`) tüm başarısız/çöken
kombinasyonların hata özeti eklendi.

**2. Colab koşumu — hata görünürlüğü işe yaradı, GERÇEK hata ortaya
çıktı:** `replay_proofs.py: error: the following arguments are required:
--mode`. Ebeveynin başlattığı `--worker` alt süreç komutu `--mode`
GÖNDERMİYORDU (worker onu hiç kullanmıyor) ama `--mode` argparse
seviyesinde `required=True` idi — ebeveyn ve işçi modlarının FARKLI
zorunlu alan kümeleri olduğundan, TEK bir argparse zorunluluk kümesi
ikisini de KARŞILAYAMADI. **Düzeltme:** `--mode` (ve zaten `SUPPRESS`
olan 9 işçiye-özgü alan) artık argparse seviyesinde `required=True`
DEĞİL — yeni `validate_args(args)` fonksiyonu `parse_args()`'tan SONRA,
`main()`'in ilk işi olarak, HER MOD için doğru zorunlulukları çalışma
zamanında denetliyor: `--worker` modunda 9 işçi alanının HEPSİ eksikse
net bir `ValueError` (hangileri eksik listeler), ebeveyn modda `--mode`
eksikse ayrı bir `ValueError`, ebeveyn modda işçiye-özgü bir alan
(yanlışlıkla) verilmişse hata VERMİYOR ama "YOKSAYILIYOR" diye
logluyor. 8 yeni test (`--worker` ile `--mode`'suz parse çalışıyor mu,
`validate_args`'ın her iki moddaki zorunluluk/yoksayma davranışı),
saf, yerelde GERÇEKTEN doğrulandı (265 passed toplam).

**3. Colab koşumu — ilk GERÇEK ispat başarılı ama 6× yavaş:**
`status=success`, `verified=True`, tepe RAM 6,4 GB, deploy gas 3.281.404,
verify gas 1.220.225, toplam gas 4.501.629 — AMA `total_ezkl_seconds=467,5`,
Faz C3'ün AYNI yapılandırma (k=1/matmul/scale=8) için ölçtüğü `~75s`'in
**6 katı**. Kod incelemesinde NEDENİ bulundu: `EZKL_TIMING_KEYS`
`"get_srs"`/`"onnx_export"`'u İÇERMİYORDU (toplam bile EKSİKTİ) VE
hiçbir ezkl adımı BİTİŞ süresini KONSOLA yazdırmıyordu — nerede
kaybedildiği GÖRÜNEMİYORDU. **Eklenen (henüz optimizasyon DEĞİL, sadece
ölçüm):** `bench_circuit.py`'nin her ezkl adımı artık kendi süresini
yazdırıyor; `round_runner.py` artık `onnx_export`'u da ölçüyor ve
fonksiyon sonunda TÜM adımların (9 adım) tek bir dökümünü basıyor;
`EZKL_TIMING_KEYS` eksiksiz. Kod okumasıyla ön-değerlendirme —
`docs/phase_e_timing_investigation.md`'de tam analiz: (a) SRS'in
yeniden indirilip indirilmediği kaynak koduyla KESİN doğrulanamadı,
bir sonraki ölçüm netleştirecek; (b) kalibrasyonun her ispatta yeniden
koşması BEKLENEN bir davranış (`param_visibility="fixed"` — farklı
ağırlıklar farklı kalibrasyon), süre farkı ölçülecek; (c) ONNX ihracının
her ispatta zorunlu olduğu (ağırlıklar ONNX'e gömülü) ama muhtemelen
ihmal edilebilir sürede olduğu değerlendirildi. Paylaşılabilirlik
tablosu: SRS (devre `logrows`'una bağlı, ağırlıklara DEĞİL) PRENSİPTE
paylaşılabilir aday; `settings.json`/`compile_circuit`/`setup`/pk-vk
HİÇBİRİ paylaşılamaz (`param_visibility="fixed"`, Faz D'nin tasarım
notu). **Optimizasyon BİLEREK uygulanmadı** — kullanıcının isteğiyle
önce gerçek per-adım verisi bekleniyor.

**4. Colab koşumu — zaman araştırması KAPANDI, YENİ bir altyapı sorunu
bulundu:** 60 ispatlık `full` mod koşusunda round 0-8 arası **36 ispat
GEÇTİ** (`verified=True`), round 9'da `requests.exceptions.ReadTimeout`
(anvil RPC yanıt vermez oldu) — ana döngü çöktü. Adım-adım kırılım
GELDİ ve zaman araştırmasını KAPATTI: `setup 36,48s + prove 37,35s ≈ 74s`
(Faz C3'ün referansıyla TAM uyumlu), `get_srs` ÖNBELLEKTEN geliyor
(0,42s) — ilk koşudaki 467,5s anomalisi SRS'in bir kerelik İLK
indirmesiydi. **ezkl tarafında optimize edilecek bir şey YOK** (bkz.
`docs/phase_e_timing_investigation.md`, "SONUÇ" bölümü). Gerçek darboğaz
anvil'in bellek birikimiydi (36 ispat × verifier deploy + calldata,
Colab'da her ispat zaten ~7GB tepe RAM yapıyor). **Düzeltmeler**
(`docs/phase_e_infra_notes.md`'de tam analiz):
- **Segment mimarisi**: `configs/schedule.yaml: replay_infra.anvil_restart_interval`
  (varsayılan 10) kadar ispat biriktiğinde, bir sonraki round'dan ÖNCE
  anvil TAMAMEN yeniden başlatılıp `RoundManager` YENİDEN deploy
  ediliyor, site'lar YENİDEN kaydediliyor (itibarlar SIFIRLANIYOR —
  "full" modda etkisi yok, "staged" modda takvim kararı script'in YEREL
  `reputations` sözlüğüne dayandığından etkilenmiyor, ama zincir-tarafı
  `reputation`/`isSiteEligible` segment başına sıfırlanıyor — bu
  AÇIKÇA loglanıp raporda not düşülüyor).
- Her round/ispat ÖNCESİ `check_rpc_alive` (`eth_blockNumber`, kısa
  zaman aşımı) sağlık kontrolü — sağlıksızsa segment yeniden başlatılıp
  (gerekirse `startRound` o round için TEKRAR çağrılıp YENİ bir
  `challengeSeed` alınarak) devam ediliyor.
- `startRound` iki denemede de başarısız olursa o round'un TÜM
  site'ları `"failed"` işaretlenip SONRAKİ round'a geçiliyor — ana
  döngü artık anvil hatasıyla ÖLMÜYOR.
- `round_fully_done`: bir round'un TÜM site'ları zaten sonuçlanmışsa
  `startRound` bile atlanıyor (resume'da gereksiz gas harcamamak için).
- `anvil --prune-history` (varsayılan 100 durum, `chain/anvil.py`) —
  İKİNCİL önlem, TEK BAŞINA yeterli olmayabileceği biliniyor
  (foundry-rs/foundry#6017), bu yüzden segment mimarisi BİRİNCİL savunma.
- `chain/client.py: Web3Client`/`RoundManagerClient` RPC timeout'u
  30s→120s (`DEFAULT_RPC_TIMEOUT_SECONDS`, `configs/schedule.yaml:
  replay_infra.rpc_timeout_seconds`'tan okunuyor) — gözlenen hatanın
  literal kaynağı buydu.
- Sonuç JSON'una `"segment_index"` (her ispat) + ayrı
  `{zk_root}/replay/replay_infra_events_<mod>.json` (TÜM yeniden
  başlatma/sağlık-kontrolü olayları + `restart_count`) eklendi —
  `docs/phase_e_replay.md`'nin yeni bir "altyapı olayları" bölümünde
  AÇIKÇA gösteriliyor, gizlenmiyor.

13 yeni saf test (`should_restart_segment`, `round_fully_done`,
`get_replay_infra_config`, `check_rpc_alive` — GERÇEK ama erişilemeyen
bir port'a karşı, mock DEĞİL — `render_infra_events`) yerelde GERÇEKTEN
doğrulandı (278 passed, 2 skipped toplam). Segment açma/kapama, gerçek
anvil restart'ı BİLİNÇLİ TEST SINIRI içinde — sadece Colab'da, gerçek
bir 60 ispatlık koşumla doğrulanabilir; henüz doğrulanmadı.

**5. Colab koşumu — "staged" mod round 0'ı geçti (sabit round, 4 site de
`verified=True`), round 1'de çöktü:** `orchestrator/schedule.py:
deterministic_unit_interval`de `AttributeError: 'int' object has no
attribute 'lower'`. Kök sebep: fonksiyon `site`'ın DAİMA bir `str`
(Ethereum adresi) olduğunu varsayıp `site.lower()` çağırıyordu, ama
`replay_proofs.py: build_round_schedule`'a `site`'ı anvil hesap
İNDEKSİ (`int`, 0-3) olarak geçiriyor. **Tasarım kararı — İKİ farklı
site kimliği temsilini TEK bir konvansiyona ZORLAMADIM:**
`orchestrator/round_runner.py`'nin canlı/üretim akışında site'ların
kanonik bir indeksi YOK, sadece kayıtlı cüzdan adresleri var (`str`
doğal); `replay_proofs.py`'nin replay akışında ise site kimliği zaten
anvil'in deterministik hesap indeksi (`int` doğal). İkisini ortak bir
temsile zorlamak birini yapay/yanlış bir veri modeline sokardı — bunun
yerine `deterministic_unit_interval`'ın kendisi `str(site).lower()` ile
tip-esnek yapıldı: `site=2` (int) ile `site="2"` (str) HER ZAMAN aynı
sonucu üretir, adresler için büyük/küçük harf farkı da elenir. Yanlış
tipte artık sessizce çökmüyor. `must_prove`/`build_round_schedule`'ın
tip belirtimleri de (`site: str` → tipsiz `site`, `sites: list[str]` →
`sites: list`) buna göre güncellendi. 3 yeni determinizm testi
(`tests/test_schedule.py`: int site çökmeden çalışıyor mu, int/str aynı
sonucu veriyor mu, aynı (seed,round,site) üçlüsü her zaman aynı değeri
veriyor mu — 5 tekrar) eklendi, yerelde GERÇEKTEN doğrulandı (281
passed, 2 skipped toplam, `tests/test_schedule.py`: 16/16).
**Madde 4 (round 0'ın zaten kaydedilmiş ispatlarının resume'da
atlanması) — kod incelemesiyle doğrulandı, DEĞİŞİKLİK GEREKMEDİ:**
`replay_proofs.py`'nin ana döngüsü `round_fully_done` kontrolünü (bir
round'un TÜM site'ları zaten `all_results`'ta varsa `startRound`'u BİLE
atlar) HER şeyden önce çalıştırıyor (satır ~703); site-bazlı
`if key in all_results and not args.force: continue` kontrolü de (satır
~730, ~757) herhangi bir chain/ezkl işleminden ÖNCE geliyor — round 0
tamamen kayıtlıysa döngü onu görmeden geçiyor.

Çıktı `{zk_root}/replay/replay_results_<mod>.json` + `docs/phase_e_replay.md`
— HER İKİSİ de script TARAFINDAN üretilir (elle yazılmadı, CLAUDE.md
madde 6 gereği gerçek veri olmadan rapor uydurulmuyor); `docs/phase_e_replay.md`
sadece o ana kadar koşulmuş mod(lar)ın GERÇEK sonuçlarını içerir, eksik
mod için "-" (tasarruf sütunu) ya da açık bir "henüz koşulmadı" notuyla
dürüstçe boş bırakılır.

**6. Colab koşumu — Faz E TAMAMLANDI, iki mod da sıfır başarısızlıkla
bitti:** tam mod 60/60 ispat (ezkl 9540,2s, solc 66,5s, deploy gas
196.880.100, verify gas 73.059.588, toplam zincir 269.939.688);
kademeli mod 21/21 ispat (ezkl 1641,9s, solc 21,6s, deploy gas
68.908.200, verify gas 25.572.981, toplam zincir 94.481.181).
Tasarruf: ispat sayısı %65, ezkl süresi %82,8, zincir maliyeti %65,0.
Kademeli takvim dağılımı: round 0/7/14 (sabit) dört site birden;
round 1 (site 1), round 4 (site 3), round 9 (site 1,3), round 12
(site 1,3), round 13 (site 0,2,3) — round 2/3/5/6/8/10/11'de hiç ispat
yok (`startRound` yine de çağrıldı, ama hiçbir site tetiklenmedi).
Tam analiz — özellikle süre tasarrufunun (%82,8) gas tasarrufundan
(%65,0) neden yüksek olduğunun kod-doğrulanmış mekanizması ve ispat
başına 2× ezkl süre farkının (159s vs 78s) araştırılan ama KESİN
kanıtlanamayan olası nedenleri — `docs/phase_e_report.md`'de.
Altyapı: her iki modda 1'er planlı anvil yeniden başlatması, sıfır
altyapı kaynaklı başarısızlık — segment mimarisi (madde 4) beklendiği
gibi çalıştı.

**7. DÜZELTME — 159s/78s farkının GERÇEK sebebi bulundu, "ortam
değişkenliği" hipotezi YANLIŞ çıktı:** kullanıcının ham
`replay_results_full.json` incelemesi ispat sürelerinin İKİ NET kümeye
ayrıldığını gösterdi — sağlıklı (`setup~36s, prove~37s`) ve bozuk
(`setup~110-160s, prove~140s`). Bozuk kümedeki TÜM 13 ispat
(round9_site0/2, round10 4 site, round11 4 site, round12_site0/2,
round13_site1) tam olarak anvil'in ÇÖKTÜĞÜ ilk tam mod koşumunda
üretilmiş — anvil'in bellek şişmesi SADECE RPC'yi değil, AYNI
makinedeki ezkl alt sürecini de (swap) yavaşlatmış. Kademeli modun 21
ispatı bu round/site aralığıyla TAM AYRIK olduğundan hiç etkilenmedi
— kademeli modun 78,185s'lik ortalaması zaten TEMİZDİ. Normalize
edildiğinde (sadece healthy verilerle, ispat başına sabit ~78,185s
üzerinden) süre tasarrufu da %65,0'a yakınsıyor — gas tasarrufuyla
BİREBİR örtüşüyor, "%82,8 > %65,0" farkı sadece bir ölçüm artefaktıymış.
**Kod:** `scripts/replay_proofs.py`'ye `classify_environment_health`/
`split_by_environment_health`/`compute_healthy_avg_ezkl_seconds`/
`render_normalized_comparison_table` eklendi (setup+prove>150s ise
"degraded"); her yeni sonuca `run_id` (main() başına üretilen, hangi
Colab koşumundan geldiğini işaretleyen) yazılmaya başlandı — eski
kayıtlarda yok, bu yüzden sınıflandırma HER ZAMAN timings üzerinden.
`docs/phase_e_report.md` iki tablo içerecek şekilde düzeltildi: (a)
ham toplamlar (bozuk veriyi içerir), (b) normalize karşılaştırma
(sadece healthy veri). Bulgu makalenin uygulanabilirlik bölümüne not
düşüldü: ZK ispat üretimi aynı makinedeki blockchain düğümünün bellek
davranışına duyarlı, kaynak izolasyonu (ayrı makine/cgroup) gerekiyor
— segment mimarisi sadece bir hafifletme, kalıcı çözüm değil. 10 yeni
saf test, yerelde GERÇEKTEN doğrulandı (291 passed, 2 skipped
toplam).

**8. DÜZELTME (nihai) — anvil bellek hipotezi de ELENDİ, gerçek sebep
Colab'ın paylaşımlı CPU değişkenliği:** madde 7'nin önerdiği 13
bozuk-ortam ispatı `--force` ile TAZE bir anvil'le (sıfır segment
restart'ı) yeniden koşuldu — sonuç İYİLEŞMEDİ, KÖTÜLEŞTİ (toplam ezkl
süresi 9949s→12173s, yeni `setup` 178-260s/`prove` 245-272s, ilk
koşumdan bile yavaş). Bu, anvil'in bellek şişmesinin bu 2-4× süre
farkının sebebi OLMADIĞININ kesin kanıtı. Devre istatistikleri
(logrows=19, num_rows=293916, pk=2517MB) 60 ispatın HEPSİNDE BİREBİR
AYNI — ağırlık/devre-boyutu hipotezi de zaten madde 6'da elenmişti.
Geriye kalan tek açıklama: Colab'ın paylaşımlı sanal makine
altyapısındaki CPU değişkenliği (yavaş ölçümler ardışık kümeler halinde
geliyor — makine durumuna işaret ediyor, ispatın kendisine değil).
**`docs/phase_e_report.md` nihai sürümüne güncellendi:** ANA TABLO
artık normalize edilmiş karşılaştırma (77,88s/ispat referansı — tam
modun 47 sağlıklı + kademeli modun 21 ispatının GERÇEK ortalaması,
68 örnek), ham toplamlar ikincil/açıklayıcı. Metodolojik not eklendi:
paylaşımlı bulut ortamlarında mutlak süre güvenilmez, makalenin ana
iddiası deterministik metriklere (ispat sayısı, gas) dayanmalı — bu
makalenin sınırlılıklar bölümüne girecek. **Makalenin raporlayacağı
sayı: %65 tasarruf, hem gas hem normalize süre için geçerli.** Ayrıca
`run_id`'nin per-kombinasyon `results/{key}.json` dosyasına HİÇ
yazılmadığı (sadece aggregate dosyaya ekleniyordu) bulunup düzeltildi
— `run_combo_in_subprocess` artık bu alanı ekleyip dosyayı yeniden
yazıyor. 13 bozuk-ortam ispatının TEKRAR koşulması ARTIK ÖNERİLMİYOR
(makine varyansına karşı geçersiz bir çözüm).

**Kabul:** maliyet karşılaştırma tablosu (`docs/phase_e_replay.md`,
script tarafından üretildi) + yorumlu analiz (`docs/phase_e_report.md`,
nihai sürüm) GERÇEK Colab verisiyle üretildi, üç koşum da (tam,
kademeli, 13 ispatın yeniden koşumu) sıfır başarısızlıkla tamamlandı,
raporlanacak sayı (%65) hem gas hem normalize süre için doğrulandı.
**KARŞILANDI.**

## Faz F (Colab) — Saldırı × koruma matrisi

**durum: kod yazıldı, Colab'da HENÜZ koşulmadı/doğrulanmadı.**

Kullanıcının bu fazı detaylandıran talimatı, PLAN.md'nin eski kısa
taslağının YERİNE geçti (bkz. `docs/phase_f_attacks.md`'nin "Tasarım
kararları" bölümü) — İKİ kapsam farkı BİLEREK not düşülüyor:
1. **LPIPS eklenmedi** — kullanıcının detaylı talimatı FID+KID+sınıf-
   tutarlılığı istedi, LPIPS'ten hiç bahsetmedi; eski taslağın LPIPS
   maddesi bu turda UYGULANMADI (istenirse ayrı bir ek olabilir).
2. **"3 round canlı entegre koşu" bu turda YAPILMADI** — ayrı bir iş
   olarak kapsam dışında bırakıldı (aşağıdaki "Kabul" bunu ayrı bir
   madde olarak işaretliyor).

`attacks/random_weights.py` (aynı şekilli TAMAMEN rastgele tensörler),
`attacks/scaled_poison.py` (ΔG=site-önceki_global, `scale × ΔG`
önceki globale eklenir — 10x/50x/100x), `attacks/conditional_poison.py`
(`mapping.embed.weight`'in DR-0/DR-4 satırlarını takas) — ÜÇÜ de SAF
state_dict operasyonu, eğitim YOK (CLAUDE.md madde 2/3/6). 14+5=19 yeni
saf test, yerelde GERÇEKTEN doğrulandı.

`attacks/detection.py`: **önemli düzeltme** — `orchestrator/round_runner.py:
generate_and_submit_proof`/`run_round` incelendiğinde, weight
commitment ve ZK devresinin HER ZAMAN AYNI `full_state` objesinden
türediği (bu kod tabanında bir sitenin "taahhüt ettiğinden FARKLI bir
ağırlığı ispatlayıp katkıda bulunması" için HİÇBİR kod yolu olmadığı)
görüldü. Yani ZK, ağırlığın İÇERİĞİNİ (iyi/kötü) DEĞİL, taahhüt-ispat-
katkı TUTARLILIĞINI kanıtlıyor — kullanıcının "conditional_poison ZK'nın
asıl değerini gösterir" beklentisi bu analizle İNCELTİLDİ: üç saldırının
DA (dürüst-ama-bozulmuş istemci modelinde) ZK tarafından
YAKALANMAMASI BEKLENİYOR (varsayılmıyor, `verify_commitment_consistency`
ile GERÇEKTEN hesaplanacak). Ayrıca ZK devresi SADECE mapping alt-ağını
kapsıyor — `random_weights`/`scaled_poison` synthesis ağını da bozduğundan
kısmen ZK'nın hiç GÖRMEDİĞİ bir alanı hedefliyor
(`attack_touches_zk_proven_scope` bunu ayırt ediyor).

`eval/metrics.py`: FID/KID StyleGAN-XL'in KENDİ `metrics.metric_main.calc_metric`'i
ÇAĞRILARAK hesaplanıyor (WebFetch ile gerçek kaynaktan doğrulandı) —
`training_options.json`'dan GERÇEK `dataset_kwargs`/`num_gpus`/`metrics`
okunuyor (`load_metric_options_from_training_options`), hiçbir ayar
uydurulmuyor. **Sınıf tutarlılığı**: kullanıcının önerdiği DR-0/DR-4
ikili KID kontrolü, TÜM DR sınıflarını (0-4) kapsayan 5×5 bir KID
karışıklık matrisine genişletildi (`class_confusion_matrix`) —
`perceived_class[g]=argmin_r kid_matrix[g][r]`, `!=g` ise swap tespiti;
StyleGAN-XL'in `kernel_inception_distance.compute_kid` formülü BİREBİR
(WebFetch ile doğrulanmış) `compute_kid_from_features` olarak taşındı,
sadece sınıf-filtrelenmiş çiftlere uygulandı (resmi fonksiyon sınıf
filtrelemeyi desteklemiyor).

`scripts/run_attacks.py`: 5 saldırı varyantı × 2 koşul (unprotected:
`fl.fedavg_utils.RunningAverage`, protected: `orchestrator.aggregate.aggregate_round`
— İKİSİ de DOĞRUDAN reuse, yeniden yazılmadı) = 10 tam pipeline.
Gerçek Faz A checkpoint'lerini (`fl.round_replay.load_site_update` +
`scripts.audit_fedavg.load_fedavg_file`) kullanır. `--metrics-mode
{full,quick}` (quick: resmi fid50k_full/kid50k_full atlanır, sadece
küçük örnekli sınıf-tutarlılığı — önce ucuz sağlama için),
`--only-attack`/`--force` ile devam edilebilir, `{zk_root}/attacks/attack_results.json`'a
atomik yazılır. 10 yeni saf test yerelde doğrulandı.

`docs/phase_f_attacks.md`: matris tablosu ŞABLONU + tasarım kararları
+ "dürüstlük noktası" (ZK içerik değil tutarlılık kanıtlar) hazır —
sayılar Colab koşumundan SONRA doldurulacak.

Tam paket: 333 passed, 2 skipped (önceki: 291).

**1. Colab koşumu — `KeyError: 'c_dim'` ile çöktü:** `scripts/run_attacks.py`,
`fl.module_tree.extract_known_attrs`'ın döndürdüğü İÇ İÇE
`{"found": {...}, "bulunamadi": [...]}` şemasını DÜZ bir sözlükmüş
gibi (`dims["c_dim"]`) okumuştu — `scripts/inventory.py`/
`scripts/make_reference_outputs.py` ise doğru şekilde
`dims["found"]["c_dim"]` kullanıyordu (grep ile TEK yanlış kullanım
yerinin `run_attacks.py` olduğu doğrulandı). **Düzeltme:** `main()`
artık `dims_result["bulunamadi"]` boş değilse net hata veriyor
(`make_reference_outputs.py`'nin deseninin AYNISI), sonra
`dims = dims_result["found"]` ile düz erişime geçiyor — kodun geri
kalanı (`dims["z_dim"]`/`dims["c_dim"]`) DEĞİŞMEDİ, artık DOĞRU
şemaya erişiyor. `tests/test_module_tree.py`'ye bu şemayı SÖZLEŞME
olarak sabitleyen yeni bir test eklendi
(`test_extract_known_attrs_return_schema_is_nested_not_flat` — üst
seviyede SADECE `found`/`bulunamadi` anahtarları olduğunu, `z_dim`/
`c_dim`'in düz erişimle bulunAMAYACAĞINI doğrular) — bu tür şema
uyumsuzluklarının gelecekte yeniden ORTAYA ÇIKMASINI önlemek için.
Tam paket: 334 passed, 2 skipped.

**2. Colab koşumu — CUDA custom op derlemesi başarısız
(`ModuleNotFoundError: No module named 'bias_act_plugin'`, ninja/`CUDA_HOME`
mevcut olmasına RAĞMEN — Colab'ın Python 3.13/güncel PyTorch
kombinasyonunun bilinen bir sorunu):** WebFetch ile gerçek kaynak
(`torch_utils/ops/bias_act.py`/`upfirdn2d.py`/`filtered_lrelu.py`)
doğrulandı — bu depoda `_init()` derleme başarısız olsa bile `False`
DÖNMÜYOR (hata `bias_act()` çağrısına kadar yükseliyor, gözlenen
hata tam BUDUR), bu yüzden `_init()`'i yamalamak yerine (StyleGAN'ın
kırılgan iç mantığına bağımlı olurdu) `eval.metrics.force_stylegan_ops_impl`
eklendi — üç modülün (`bias_act`/`upfirdn2d`/`filtered_lrelu`) genel
işlevini sarmalayıp `impl` argümanını HER ÇAĞRIDA zorluyor (internal
katman kodu `impl=` kwarg'ını hiç geçmiyor, hep `'cuda'` varsayılanına
güveniyor). `scripts/run_attacks.py`'ye `--ops-impl {auto,cuda,ref}`
eklendi — `auto` (varsayılan) KÜÇÜK bir deneme üretimiyle CUDA'yı
test edip BAŞARISIZ olursa UYARIYLA `ref`e düşüyor (sessizce
geçmiyor), kullanılan mod sonuç JSON'una (`ops_impl` alanı) kaydediliyor.
`class_confusion_matrix` artık `generation_seconds`/`num_generated_images`
de döndürüyor — `estimate_full_mode_cost` bunlardan GERÇEK ölçülen
saniye/görüntü ile `--metrics-mode full`'un (fid50k_full+kid50k_full,
50k görüntü × 10 koşul) tahmini maliyetini (saat) hesaplayıp `ref`
modundaysa UYARI olarak logluyor — 74s gibi bir sayı uydurulmadı,
her koşumda TAZE hesaplanıyor. `docs/phase_f_attacks.md`'ye
`13.13` referansıyla karşılaştırılabilirlik sınırlılığı (ref modunun
cuda ile matematiksel eşdeğerliği bu oturumda sayısal DOĞRULANAMADI)
not düşüldü. 6 yeni saf test (`_force_kwarg_wrapper`×3,
`estimate_full_mode_cost`×3) yerelde doğrulandı. Tam paket: 340
passed, 2 skipped.

**3. Colab koşumu — `ref` moduna geçince CUDA belleği tükendi
(`upfirdn2d._upfirdn2d_ref`'in `F.pad`'i TEK görüntü için 13.37 GiB
istedi, 39.49 GiB'lık GPU'da):** StyleGAN3-r'ın geniş filtreleriyle
`ref` yolunun (unfused) TEK bir ileri geçişte bile devasa ara tensör
üretmesi — CUDA'nın fused kernellerinin TEK işlemde yaptığını `ref`
birden fazla büyük tensörle yapıyor. Düzeltmeler: `--gen-batch-size`
(varsayılan `ops_impl`'e göre otomatik, `ref`→1/`cuda`→32);
`eval.metrics.generate_class_images` artık OOM'u (`is_cuda_oom_error`
— yeni `torch.cuda.OutOfMemoryError` + eski `RuntimeError` mesaj
tabanlı, PyTorch sürümü varsayılmıyor) yakalayıp batch boyutunu
YARIYA indirip yeniden deniyor (4 kez), batch 1'de bile OOM olursa
net hata + `--device cpu` önerisiyle duruyor; her batch sonrası
`del`+`gc.collect()`+`torch.cuda.empty_cache()`; üretim zaten
`torch.no_grad()` içindeydi. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
`torch` import edilmeden ÖNCE ayarlanıyor. `--device {auto,cuda,cpu}`
eklendi — `cpu`ysa `resolve_ops_impl` smoke-testi atlayıp doğrudan
`ops_impl='ref'` dönüyor (StyleGAN-XL'in `impl` dallanması CPU
tensöründe zaten HER ZAMAN ref yoluna düşüyor), CPU süresi de
`estimate_full_mode_cost` ile raporlanıyor. **Değerlendirilip GEREKSİZ
bulunan önlem:** "4 site + shell GPU'da mı duruyor" sorusu kod
incelemesiyle netleştirildi — `load_network_pkl` hiç `.cuda()`
çağırmıyor, ağırlıklar zaten CPU'da; OOM'un kaynağı BİRİKEN kopyalar
DEĞİL, `ref` yolunun TEK geçişteki ara tensör boyutuydu — ek bir
"CPU'ya taşı" değişikliği GEREKMEDİ. 9 yeni saf test
(`is_cuda_oom_error`×4, `pick_default_gen_batch_size`×3, CLI×2)
yerelde doğrulandı. Tam paket: 349 passed, 2 skipped.

**4. Colab koşumu — quick mod 10/10 geçti, KRİTİK bulgu + bir kod
hatası daha bulundu:**

**Kritik bulgu (GERÇEK, makalenin ana tezi):** `||ΔG||` — random_weights
59592, scaled_poison_10x/50x/100x 25066/125332/250664 (hepsi
`tau=3000`'i AÇIKÇA aşıyor, norm kontrolü yakalıyor) — ama
**`conditional_poison`: `||ΔG||=2507 < tau=3000`, norm kontrolü
KAÇIRIYOR.** `tau` keyfi değil (Faz A'nın gerçek p99=2518 ölçümünden)
ama bu saldırı TAM OLARAK o meşru aralığın içinde kalacak şekilde
inşa edilebiliyor — mevcut norm+ZK savunma katmanı bu sınıf saldırıya
KÖR. `zk_caught` beş saldırının BEŞİ için de `False` — bu Colab'a
bağlı değil, `verify_commitment_consistency`'nin bir state'i
kendisiyle karşılaştırmasından doğan KOD GARANTİSİ (matematiksel
kesinlik, ayrıca doğrulanması gerekmiyor).

**Bulunan kod hatası:** `--ops-impl auto` smoke-test'i
`RuntimeError: Expected all tensors to be on the same device (...
wrapper_CUDA__index_select)` ile çöktü — bu bir CUDA DERLEME hatası
DEĞİLDİ, `resolve_ops_impl` `z`/`c`'yi `device`'a taşırken `g_ema`'nın
KENDİSİNİ hiç taşımıyordu, eski kod bunu geniş `except Exception` ile
yakalayıp "derleme başarısız" diye YANLIŞ teşhis edip SESSİZCE ref'e
düşüyordu — CUDA GERÇEKTEN çalışıyor olabilirdi ama hiç test
EDİLEMEMİŞTİ. **Düzeltme:** `resolve_ops_impl` artık smoke-test'ten
ÖNCE `g_ema.to(device)` çağırıp `eval.metrics.assert_module_on_device`
(TÜM parametre+buffer'ları tek tek kontrol eder) ile doğruluyor;
`is_device_mismatch_error` bu SINIF hatayı gerçek derleme
hatalarından AYIRT edip AYRI, net bir hata olarak yükseltiyor (bir
daha sessizce yanlış teşhis edilmiyor). **CUDA'nın bu düzeltmeyle
GERÇEKTEN çalışıp çalışmadığı henüz YENİDEN test EDİLMEDİ** — bir
sonraki Colab koşumu gösterecek.

**Pratiklik sorunu + çözüm:** ölçülen `0.498 s/görüntü` (ref, quick
mod) ile `estimate_full_mode_cost` `--metrics-mode full`'un TOPLAM
~138 saat süreceğini gösterdi — yapılamaz. `--metrics-mode custom
--fid-num-gen <N>` (varsayılan 5000) eklendi —
`eval.metrics.run_custom_fid_kid` StyleGAN-XL'in KENDİ
`frechet_inception_distance.compute_fid`/`kernel_inception_distance.compute_kid`
fonksiyonlarını DOĞRUDAN çağırıp SADECE resmi sarmalayıcının
hardcoded `num_gen=50000`'ini atlıyor (mantık yeniden yazılmıyor);
sonuç `fid_comparable_to_reference=False` alanıyla VE her konsol
çıktısında "13.13 İLE KARŞILAŞTIRILAMAZ" notuyla işaretleniyor.

**Rapor çerçevesi (madde 4):** FID kaba saldırıları (zaten norm
kontrolüyle yakalanan) gösterir; sınıf-tutarlılığı matrisi
`conditional_poison`'ın FID'de GÖRÜNMEYEN asıl zararını gösterir —
`docs/phase_f_attacks.md` bu ayrıma göre yeniden çerçevelendi, GERÇEK
norm/ZK sayılarıyla dolduruldu.

**Tamamlanamayan (madde 5):** `attack_results.json`'a bu oturumda
erişim yok (Colab/Drive'da, yerel makineye senkron değil) —
`conditional_poison`'ın sınıf-tutarlılığı matrisinin GERÇEKTEN DR-0/DR-4
takasını gösterip göstermediği bu dosyanın (en azından
`class_confusion` alanlarının) paylaşılmasını bekliyor; doküman bunu
"Bekleyen veri" olarak AÇIKÇA işaretliyor, uydurmadı.

7 yeni saf test (`assert_module_on_device`×2, `is_device_mismatch_error`×3,
CLI×2) yerelde doğrulandı. Tam paket: 356 passed, 2 skipped.

**5. Colab koşumu — `conditional_poison`'ın sınıf-tutarlılığı matrisi
geldi, takas GÖRÜNMÜYOR (yeni araştırma):** unprotected/protected
BİREBİR AYNI (BEKLENEN — koruma zaten bu saldırıyı durduramadı, madde
4). Ama matrisin kendisi: istenen DR-0 → en yakın gerçek DR-0
(0.0489, beklenen DR-4); istenen DR-4 → en yakın gerçek DR-1 (0.0599,
beklenen DR-0) — köşegen BÜYÜK ÖLÇÜDE korunmuş, takas GÖRÜNMÜYOR.
Üç hipotez sırayla test ediliyor: **H1** FedAvg seyreltmesi (zehirli
site 1/4 ağırlıkla katkıda bulunuyor), **H2** gömme takası yetersiz
(sınıf bilgisi embed_proj/fc0/fc1'den de akıyor olabilir), **H3** KID
ayrım gücü yetersiz (quick modun 20 görüntü/sınıf'ı az, DR sınıfları
zaten görsel olarak yakın). Kullanıcının talimatıyla SADECE H1 testi
yazıldı: `scripts/run_attacks.py`'ye yeni bir tanı koşulu eklendi —
**`poisoned_alone`** (`CONDITIONS` artık 3 eleman) —
`build_poisoned_alone_global` FedAvg'ı TAMAMEN ATLAYIP zehirli
site'ın KENDİ ağırlığını (seyreltme yok) doğrudan "global" olarak
kullanıp aynı FID+sınıf-tutarlılığı işlem hattından geçiriyor, TÜM
saldırılar için otomatik hesaplanıyor. **Önerilen güçlendirme (H1
doğrulanırsa)**: birden fazla kötücül site EKLEMEK yerine (tehdit
modelini değiştirir), `scaled_poison`'ın ΔG-ölçekleme mantığının
AYNISI `conditional_poison`'a uygulanabilir —
`poisoned_row = 4*target_row - 3*honest_row` (FedAvg SONRASI ortalama
TAM hedefe ulaşacak şekilde kompanse edilmiş katkı) — bu kod HENÜZ
YAZILMADI, H1 sonucunu bekliyor. 3 yeni saf test
(`build_poisoned_alone_global`×2, `CONDITIONS` içeriği×1) yerelde
doğrulandı. Tam paket: 359 passed, 2 skipped.

**6. H1 DOĞRULANDI (`poisoned_alone`):** DR-4 satırında gerçek DR-0'a
KID=0.0127 vs gerçek DR-4'e 0.0710 (`perceived_class[4]=0`) — takas
tek başına AÇIKÇA görünüyor; FedAvg sonrası aynı hücre 0.0667'ye çıkıp
kayboluyor (1/4 seyrelme). H2/H3'e gerek kalmadı. Gözlem: DR-0 tarafı
asimetrik (`perceived_class[0]=1`), DR-0 sütunu tüm satırlarda yüksek —
ölçüm sınırlılığı, kanıt DR-4 satırı. **Eklenen:**
`attacks/conditional_poison.py: compensated_swap_embed_rows`
(`poisoned_row = N*target − (N−1)*honest`, N=`NUM_SITES`=4; FedAvg sonrası
ortalama TAM hedefe ulaşır, birim testle doğrulandı; embed delta'sı basit
takasın tam N katı) ve yeni saldırı `conditional_poison_compensated`
(`conditional_poison` KORUNDU — karşılaştırma için). `run_attacks.py` her
saldırı için `natural_delta_norm`/`attack_only_delta_norm`'u da
loglayıp JSON'a yazıyor (kompanzasyonun `tau=3000`'i aşıp aşmadığı GERÇEK
sayıyla Colab'da görülecek — aşarsa norm kapısı yakalar, "ince saldırı
geçer" tezi sadece basit versiyona dayanır). `resolve_ops_impl`
fallback'te TAM traceback basıp `ops_impl_fallback_error` olarak JSON'a
yazıyor (gerçek mesaj Colab logundan bekleniyor). Testler: 367 passed,
2 skipped.

**Kabul (İKİ ayrı madde):**
1. Saldırı × koruma matrisi (bu bölüm) — **HENÜZ KARŞILANMADI**, Colab
   koşumu bekleniyor.
2. 3 round canlı entegre koşu — **bu turda hiç ELE ALINMADI**, ayrı
   bir istek/tur gerektirir.
