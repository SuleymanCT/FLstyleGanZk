# Faz C3 Raporu — Tam Benchmark Izgarası (30 Kombinasyon)

**Durum: Faz C TAMAMLANDI.** Bu rapor `scripts/bench_circuit.py`'nin
Colab'daki tam ızgara koşumunun (k ∈ {1,4,8} × embed_mode ∈
{matmul,gather} × scale ∈ {6,7,8,9,10} = 30 kombinasyon) ve ardından
gelen iki hedefli doğrulama turunun (bkz. "Üç hata ve düzeltmeleri")
gerçek çıktısına dayanır. `docs/phase_c_calibration.md` (kalibrasyon
teşhisi) bu raporun ön koşuludur.

## Operasyonel yapılandırma ve ölçülen değerler (k=1, matmul, scale=8)

Nihai doğrulama koşumunda (`--only k1_matmul_scale8`) ölçülen, üretim
yapılandırmasının (`configs/circuit.yaml`) tam sonucu:

| Metrik | Değer |
|---|---:|
| challenge_size_k | 1 |
| embed_mode | matmul |
| scale (input_scale=param_scale) | 8 |
| setup süresi | 37.3 s |
| prove süresi | 37.4 s |
| tepe RAM | 7.2 GB |
| proof boyutu | 96 KB |
| verifier deployed bytecode | 14.929 B (EIP-170'in %60,7'si) |
| verify_gas (zincir üstü, DOĞRULANDI) | 1.174.788 |
| max_abs_error (bağıl) | %3,21 |
| decomposition uyarısı | 0 |
| durum | `success`, zincir üstü doğrulama `verified=True` |

Bu tablo, aşağıdaki bölümlerin dayandığı ilk (30 kombinasyonluk) koşumun
sayılarından ufak farklar (ör. prove 37.1s→37.4s, verify_gas
1.174.812→1.174.788) taşıyabilir — ayrı bir Colab VM koşumunun doğal
değişkenliği (gas/zamanlama, nonce vb.), bulguları DEĞİŞTİRMİYOR.

## Özet: ölçek tavanı, embed_mode farkı, EIP-170 sınırı

1. **Ölçek tavanı = 8**, her `k`/`embed_mode`'da tutarlı. `scale=6,7,8`
   HER kombinasyonda kalibrasyonu geçti; `scale=9` ve `scale=10` HER
   kombinasyonda kalibrasyonda düştü. `docs/phase_c_calibration.md`'nin
   ince taramayla ({6,7,8,9,10}) doğruladığı sınır tam olarak buydu.
2. **`decomposition_uyari` her kombinasyonda 0.** Faz B'nin oyuncak
   modelinde görülen "decomposition error: integer ... is too large"
   uyarıları (fatal değil) gerçek mapping devresinde HİÇ görülmedi —
   bunun yerine scale 9/10'da doğrudan FATAL bir kalibrasyon hatası
   çıkıyor (`docs/phase_c_calibration.md` Bulgu 2). Yani gerçek devre,
   oyuncak modelin "yavaşça bozulma" davranışı yerine sert bir eşikte
   çalışıp durduruyor.
3. **`embed_mode="matmul"`, `"gather"`'dan HER metrikte kabaca 2× daha
   ucuz** (aşağıdaki tablo). ArgMax/Gather'sız olmasının (ezkl güvenliği)
   yanı sıra performans açısından da AÇIK ARA daha iyi.
4. **EIP-170: sadece k=1 geçiyor.** k=4/k=8 verifier'ları deploy
   edilemeyecek kadar büyük — protokol tasarımı buna göre şekillendi
   (aşağıda "Protokol tasarımına etki" bölümü).
5. **scale=8'de bağıl hata ~%3.2-3.9** (k'ye göre değişiyor) — aşağıdaki
   "Fidelity: max_abs_error" bölümü.

## matmul vs gather (k=1, scale=8) — ikisi de zincir üstünde doğrulandı

| Metrik | matmul | gather | oran (gather/matmul) |
|---|---:|---:|---:|
| setup süresi | 36.0 s | 69.6 s | 1.93× |
| prove süresi | 37.1 s | 75.5 s | 2.04× |
| pk boyutu | 2.52 GB | 5.61 GB | 2.23× |
| tepe RAM | 7.3 GB | 17.3 GB | 2.37× |
| verifier deployed bytecode | 14.929 B | 19.735 B | 1.32× |
| verify_gas | 1.174.812 | 1.350.486 | 1.15× |

**Yorum:** `matmul` sadece "ArgMax/Gather yerine MatMul" gibi kozmetik
bir devre optimizasyonu değil — gerçek devrede pk boyutu ve tepe RAM'de
~2-2.4× fark yaratıyor. Bunun muhtemel sebebi: `gather` modundaki
`ArgMax`+`Gather`+`Cast` zincirinin ezkl tarafında ayrık işlemler için
gereken EK kısıt/lookup yapılarını (seçim/indeksleme devresi) devreye
sokması — `matmul` bunun yerine devrenin zaten sahip olduğu sıradan
`MatMul` altyapısını kullanıyor, ek bir yapı gerektirmiyor. Bu, C2'nin
sonunda öngörülen "makalede somut bir devre optimizasyonu bulgusu
olacak" beklentisini ölçülebilir sayılarla doğruluyor.

## EIP-170: deployed bytecode ve aşım durumu

| k | embed_mode | deployed bytecode (B) | EIP-170 (24.576 B) | zincir üstü doğrulama |
|---|---|---:|---|---|
| 1 | matmul | 14.929 | ✅ altında | ✅ mümkün, DOĞRULANDI |
| 1 | gather | 19.735 | ✅ altında | ✅ mümkün, DOĞRULANDI |
| 4 | matmul | 26.757 | ❌ AŞIYOR | ❌ deploy edilemiyor |
| 4 | gather | 26.758 | ❌ AŞIYOR | ❌ deploy edilemiyor |
| 8 | matmul | 32.144 | ❌ AŞIYOR | ❌ deploy edilemiyor |
| 8 | gather | 32.142 | ❌ AŞIYOR | ❌ deploy edilemiyor |

k=4/k=8'de `embed_mode`'un bytecode boyutuna etkisi neredeyse sıfır
(26.757 vs 26.758, 32.144 vs 32.142) — devre boyutu bu ölçekte `k`
(kaç örneğin aynı anda ispatlandığı) tarafından domine ediliyor,
`embed_mode` farkı k=1'deki kadar belirgin değil. Bu da makul: `k`
arttıkça mapping ağının TÜM ileri geçişi `k` kez tekrarlanıyor
(devre `k` katına çıkıyor), embed seçim stratejisinin payı görece
küçülüyor.

`docs/phase_b_report.md`'deki "Halo2/ezkl verifier boyutu kısıt
sayısıyla kabaca doğrusal büyür" öngörüsü ve `PLAN.md`'nin Faz B'den
gelen notu ("EIP-170 aşma ihtimali gerçek") burada TAM olarak
doğrulandı — sadece k=1 sınırın altında kalıyor.

## Fidelity: `max_abs_error` (scale=8)

| k | embed_mode | max_abs_error | max_abs_error_% (referans `w`'ye oranla) |
|---|---|---:|---:|
| 1 | matmul | 0.1331 | %3,21 |
| 4 | matmul | 0.137 | %3,205 |
| 8 | matmul | 0.1658 | %3,879 |

(`docs/phase_c_calibration.md`'nin ilk teşhis koşumunda k=1 için
gözlenen `0.133` ile aynı büyüklük mertebesinde — sağlıklı, çünkü
kalibrasyon tarafı k'den etkilenmiyor, farklar sadece hangi örneklerin
w çıktısına düştüğünden kaynaklanıyor.)

**Bu ispatın GEÇERLİLİĞİNİ etkilemiyor.** ezkl'nin ürettiği ispat,
"belirli bir SABİT NOKTALI (fixed-point, scale=8) kuantize devrenin
belirli bir tanık için doğru şekilde hesaplandığı"nı kanıtlıyor — kanıt
sistemi kuantize aritmetiğin KENDİSİNİ doğruluyor, orijinal fp32 mapping
ağının çıktısını DEĞİL. Ağırlıkların gerçek (fp32) StyleGAN-XL
checkpoint'inden geldiği iddiası ayrı bir mekanizmayla (shard'ın
`canonical_hash`'i / taahhüt) bağlanıyor, ispatın kendisi bu bağlamayı
kanıtlamıyor. Dolayısıyla `max_abs_error≈%3,2-3,9`, kuantizasyonun
FARKLILAŞTIRABİLİR görüntü çıktısı üzerindeki etkisini gösteren, C4'te
(kuantizasyon etkisi — kosinüs benzerliği/L2 farkı, DR sınıfı başına)
daha ayrıntılı ele alınacak AYRI bir soru; ama makalede scale seçiminin
maliyetini (daha yüksek fidelity istenirse scale artırılabilir, ama bu
raporun gösterdiği gibi mevcut devre boyutunda scale=8 ÜST SINIR)
göstermek için raporlanması gereken somut bir sayı.

## Protokol tasarımına etki: challenge başına k=1

Bu bulgunun doğrudan sonucu: **on-chain ispat protokolü (Faz D) HER
challenge turunda tek bir (z, c) örneği ispatlamalı, birden fazla
örneği tek bir devrede toplu (batch) ispatlayamaz** — k=4/k=8
verifier'ları zaten deploy edilemiyor, dolayısıyla bu yönde bir
"toplu ispat" tasarımı bu devre mimarisiyle MÜMKÜN DEĞİL. `configs/circuit.yaml`
bu yüzden `challenge_size_k: 1` olarak sabitlendi (bkz. dosyadaki
yorum). Faz D'nin `RoundManager.sol` tasarımı, bir round içindeki
her doğrulama talebini AYRI bir `submitProof` çağrısı olarak ele almalı
— tek bir çağrıda birden fazla örneği aynı anda kanıtlayan bir akış
kurulamaz.

## Üç hata ve düzeltmeleri

### 1. `ezkl.deploy_evm` — iki aşamalı hata (private key formatı, sonra argüman sırası)

k=4/k=8 kombinasyonlarında **eskiden** `compile_verifier_solidity`
BAŞARILI oluyordu (bytecode boyutunu zaten biliyoruz: 26.757/26.758,
32.144/32.142) ama hemen ardından `deploy_and_verify_onchain`
(`Web3Client.deploy_bytecode`) EIP-170 yüzünden DOĞAL OLARAK
başarısız oluyordu (`receipt.status != 1`) — bu `RuntimeError`, solc
derlemesiyle AYNI `except RuntimeError` bloğuna düştüğü için (mimari
hata: ikisi aynı `try` içindeydi) "solc tamamen başarısız" sanılıp
gereksiz yere `deploy_and_verify_via_ezkl_native` (ezkl'nin kendi
`deploy_evm`'i) fallback'ine düşülüyordu. Orada 1. tur şu hatayı
verdi:

```
RuntimeError: Failed to run deploy_evm: [eth] Private key must be in
hex format, 64 chars, without 0x prefix
```

(`chain.anvil.AnvilProcess`'in verdiği private key `0x` ÖNEKLİ —
`Web3Client`/web3.py'nin beklediği format — ama `ezkl.deploy_evm`
önekSİZ, tam 64 hex karakter bekliyor.) Bu `chain/anvil.py:
normalize_private_key_hex` ile düzeltildi. Hedefli yeniden koşumda
(`--only k4_matmul_scale8,k8_matmul_scale8 --force`) bu kez **2. bir
hata** çıktı:

```
RuntimeError: Failed to run deploy_evm: [eth] failed to parse url
/content/zk_bench_work/k4_matmul_scale8/Verifier.sol
```

`sol_code_path`'in DEĞERİ `rpc_url` parametresi olarak yorumlanmış.
`ezkl.pyi`'nin (v23.0.5 etiketi) belgelediği pozisyonel sıra
(`addr_path, sol_code_path, rpc_url, contract_type, optimizer_runs,
private_key`) ile GERÇEK pyo3 bağlamasının kabul ettiği sıra FARKLIYMIŞ
— yani dokümantasyon (bu noktada) güvenilir bir referans değilmiş.
**Kalıcı düzeltme:** `deploy_and_verify_via_ezkl_native` artık
`ezkl.deploy_evm`'i POZİSYONEL DEĞİL, `ezkl.pyi`'den alınan parametre
ADLARIYLA (kwarg) çağırıyor — hangi pozisyonel sırayı kabul ettiği
artık önemsiz.

**Asıl mimari düzeltme:** EIP-170 aşımı artık AYRI ele alınıyor. `solc`
derlemesi BAŞARILI olup `exceeds_eip170=True` çıkarsa, deploy HİÇ
DENENMİYOR — ne `Web3Client` ne `ezkl.deploy_evm` çağrılıyor — sonuç
doğrudan `status="eip170_exceeded"` (`"failed"`den AYRI bir durum)
olarak işaretleniyor, log'a net bir "EIP-170 AŞIMI: ... deploy HİÇ
DENENMİYOR" satırı yazılıyor. `deploy_and_verify_via_ezkl_native`
fallback'i artık SADECE solc'un GERÇEKTEN (boyuttan bağımsız bir
sebeple) tamamen başarısız olduğu durumda tetikleniyor.

**DOĞRULANDI (`--only k4_matmul_scale8,k8_matmul_scale8 --force`):**
her iki kombinasyon da temiz bir şekilde `status="eip170_exceeded"`,
`error_summary=None` verdi — ne private-key ne url-parse hatası, deploy
hiç denenmedi. Beklenen davranış TAM olarak gerçekleşti.

**NOT — k4/k8'in `gather` varyantları YENİDEN koşulmadı**, `bench_results.json`'da
hâlâ eski (private-key hatalı) `"failed"` durumunda duruyorlar. Bunun
sonucu ETKİLEMEYECEĞİNİ biliyoruz: `matmul` ile aynı k'de bytecode boyutu
neredeyse özdeş (26.758/32.142 vs 26.757/32.144 — yukarıdaki "EIP-170:
deployed bytecode ve aşım durumu" tablosu) ve zaten EIP-170'i AŞIYOR,
yani yeniden koşulsalar da sonuç aynı şekilde `eip170_exceeded` olurdu.
İstenirse `--only k4_gather_scale8,k8_gather_scale8 --force` ile ayrıca
doğrulanabilir, ama Faz C'nin kapanışını beklemiyor.

### 2. `max_abs_error`/`max_abs_error_%` tabloda boştu

`extract_max_abs_error`'un eski regex'i basit `max_abs_error=X` biçimini
arıyordu; ezkl'nin GERÇEK "Numerical Fidelity Report"u ise ÇOK SÜTUNLU
bir pipe'lı markdown tablo (başlık satırı + veri satırı, `|` ile
ayrılmış — `mean_error | max_error | max_abs_error | ...` gibi).
**Düzeltme:** `parse_markdown_table_column` (yeni, saf/testable) bir
başlık satırında `column_name`'i bulup aynı sütun indeksindeki değeri
sıradaki veri satırından okuyor; `extract_max_abs_error` önce bunu
dener, bulamazsa eski basit biçime düşüyor (geriye uyumlu). Log'da
"max_abs_error" GEÇTİĞİ HALDE yapısal ayrıştırma yine de başarısız
olursa (ör. hiç beklenmeyen üçüncü bir biçim), sonuç sessizce boş
bırakılmıyor — `"yakalanamadi"` olarak AÇIKÇA işaretleniyor (`-` = rapor
hiç yok, `"yakalanamadi"` = rapor var ama ayrıştırılamadı; bu ikisi
karışmasın diye ayrı).

### 3. `gerceklesen_scale` tabloda boştu

Kaynak: basit bir programlama hatası. `run_ezkl_pipeline`
`realized_scale`/`requested_scale`'i doğru hesaplayıp DÖNDÜRÜYORDU, ama
`run_worker` bu iki alanı `setup_result`'tan `result`'a KOPYALAMAYI
UNUTMUŞTU (`result["pk_size_bytes"]`/`vk_size_bytes` kopyalanıyordu,
`realized_scale`/`requested_scale` değil). Şema/kalibrasyon sorunu
DEĞİLDİ — `read_realized_scale`'in kendisi `docs/phase_c_calibration.md`'nin
teşhis koşumunda zaten doğru çalıştığı kanıtlanmıştı. **Düzeltme:**
`run_worker`'a eksik iki satır eklendi.

## Ek: `--only` bayrağı

`scripts/bench_circuit.py --only k4_matmul_scale8,k1_matmul_scale8` artık
SADECE belirtilen kombinasyonları (virgülle ayrılmış `combo_key`
biçiminde) yeniden koşturuyor — `--k-values`/`--embed-modes`/`--scales`/
`--only-first` yoksayılıyor. Yukarıdaki üç düzeltmeyi doğrulamak için
30 kombinasyonun tamamını baştan koşmaya gerek yok; `--force` ile
birlikte kullanılıp sadece etkilenen kombinasyonlar (k=4/k=8'in deploy
adımı, ve max_abs_error/realized_scale'in HER kombinasyonda yeniden
doğrulanması gereken durumlarda hepsi) yeniden çalıştırılabilir.

## Operasyonel karar: `configs/circuit.yaml`

`challenge_size_k=1`, `embed_mode=matmul`, `scale=8` olarak sabitlendi
(gerekçeler dosyanın kendi yorumlarında ve yukarıda). Bu, Faz D'nin
kontrat/orkestratör tasarımının üzerine kurulacağı sabit devre
parametre kümesi.
