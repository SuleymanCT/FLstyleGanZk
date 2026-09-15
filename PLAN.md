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

**durum: kod yazıldı, Colab'da HENÜZ koşulmadı/doğrulanmadı.**

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
`render_mode_comparison_table`, `render_staged_distribution`) 13 testle
yerelde GERÇEKTEN doğrulandı (254 passed, 2 skipped toplam). ezkl/anvil/
solc/gerçek shard gerektiren kısımlar (`run_worker`, `main`'in chain
dalı) BİLİNÇLİ TEST SINIRI içinde — sadece Colab'da doğrulanabilir.

Çıktı `{zk_root}/replay/replay_results_<mod>.json` + `docs/phase_e_replay.md`
— HER İKİSİ de script TARAFINDAN üretilir (elle yazılmadı, CLAUDE.md
madde 6 gereği gerçek veri olmadan rapor uydurulmuyor); `docs/phase_e_replay.md`
sadece o ana kadar koşulmuş mod(lar)ın GERÇEK sonuçlarını içerir, eksik
mod için "-" (tasarruf sütunu) ya da açık bir "henüz koşulmadı" notuyla
dürüstçe boş bırakılır.

**Kabul:** maliyet karşılaştırma tablosu üretildi. **HENÜZ KARŞILANMADI**
— Colab'da önce `--mode full --only-first`, sonra tam `--mode full` ve
`--mode staged` koşulup gerçek `docs/phase_e_replay.md` üretilene kadar
bu faz "tamamlandı" sayılmayacak.

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
