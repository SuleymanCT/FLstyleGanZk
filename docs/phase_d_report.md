# Faz D Raporu — Kontratlar, Orkestratör, Zincir Üstü Doğrulama

**Durum: Faz D TAMAMLANDI.** `contracts/RoundManager.sol`, GERÇEK ezkl
ispatlarıyla, anvil üzerinde, `tests/test_contracts.py`'nin altı
senaryosunun TAMAMIYLA (6/6) Colab'da doğrulandı. Bu rapor mimari kararı
(ve gerekçesini), test kapsamını, ölçülen değerleri ve yol boyunca
çözülen dört veri-biçimi tuzağını belgeliyor — sonuncusu, ezkl+web3
entegrasyonu yapacak herkesin muhtemelen karşılaşacağı, genellenebilir
bulgular.

## Mimari karar: verifier-per-proof (tek sabit Verifier YOK)

`RoundManager.sol` kullanıcının orijinal spesifikasyonundan BİLEREK
sapıyor: TEK bir sabit `verifier` adresi tutmuyor, `submitProof(roundId,
verifierAddress, proof, publicInputs)` verifier adresini PARAMETRE
olarak alıyor.

**Gerekçe:** Faz B/C boyunca ezkl `param_visibility="fixed"` kullanıldı
— ağırlıklar devrenin SABİT sütunlarına (verifier bytecode'unun İÇİNE)
gömülü. Federe öğrenmede her round/site'ın ağırlıkları FARKLI
olduğundan, HER (round,site) ispatının KENDİ verifier'ı olmak ZORUNDA —
tek bir sabit adres round 2'den itibaren YANLIŞ olurdu (round 0'ın
verifier'ı, round 7'nin ağırlıklarını asla doğrulayamaz). `orchestrator/round_runner.py`
bu yüzden her GEREKLİ ispat için TAZE bir Verifier deploy edip adresini
`submitProof`'a veriyor; kontrat bunu `Submission.verifierUsed`'da
kaydedip sonradan denetlenebilir kılıyor.

**Somut maliyet sonucu:** HER gerekli ispat artık ezkl setup+prove
(~75s, Faz C3 ölçümü: k=1/matmul/scale=8 için setup 37.3s + prove
37.4s) + solc derleme (~20s, Faz B ölçümü) + verifier deploy gas'ı +
`submitProof` gas'ı gerektiriyor. Bu, `orchestrator/schedule.py`'nin
`random_ratio=0.3` örneklemesinin (her round her site yerine) neden
EKONOMİK OLARAK GEREKLİ olduğunu somutlaştırıyor — protokol her round
her site'ı ispatlasaydı, 15 round × birden fazla site için toplam
maliyet hızla biriktirirdi.

## Test kapsamı — 6/6, GERÇEK ezkl ispatlarıyla

`tests/test_contracts.py`, Drive/pkl gerektirmeyen küçük bir SENTETİK
mapping devresiyle (aynı `circuits.rebuild_mapping`/`circuits.export_mapping`/
`scripts.bench_circuit` fonksiyonları, gerçek k=1/matmul devresiyle
AYNI ezkl setup/prove/verify zincirinden geçerek) kuruldu — bu, testi
Colab'a/Drive'a özgü olmaktan çıkarıp hızlandırırken yine de GERÇEK
bir ezkl ispatı ve GERÇEK bir Verifier.sol üzerinden çalışıyor.

Colab'da doğrulanan altı senaryo:

| Senaryo | Sonuç |
|---|---|
| Mutlu yol: `startRound` → `submitUpdate` → `submitProof` → `finalizeRound` | ✅ geçerli ispat kabul edildi, itibar arttı, round kapandı |
| Geçersiz ispat (tek bit bozulmuş, GERÇEK bir ispattan türetilmiş) | ✅ reddedildi, itibar `-reputationPenalty` düştü |
| İtibar eşiğin altına düşen site | ✅ `isSiteEligible=false`, dışlandı |
| Aynı round'a çift `submitUpdate` | ✅ reddedildi (`"RoundManager: bu round'a zaten gonderim yapildi"`) |
| Kayıtsız site'nin `submitUpdate`'i | ✅ reddedildi (`"RoundManager: kayitli site degil"`) |
| Doğrulanmamış site'nin `finalizeRound`'a dahil edilmesi | ✅ reddedildi (`"RoundManager: dogrulanmamis site finalizeRound'a dahil edilemez"`) |

Kritik doğrulama: `IHalo2Verifier.verifyProof(bytes,uint256[]) returns
(bool)` imza varsayımı GERÇEK ezkl Verifier'ıyla TUTTU —
`RoundManager.submitProof`'un `try/catch`'i hiçbir zaman "güvenli
başarısızlık" (revert yakalama) dalına düşmeden, GERÇEK `bool` dönüş
değeri üzerinden çalıştı. `try/catch` yine de savunma katmanı olarak
kod tabanında kalıyor (bkz. `contracts/RoundManager.sol`'un güncellenmiş
yorumu).

## Ölçülen değerler

| Metrik | Değer | Kaynak |
|---|---:|---|
| RoundManager deployed bytecode | 6.867 B | Faz D, bu koşum |
| Verifier deployed bytecode (test devresi) | 14.921 B | Faz D, bu koşum |
| Verifier deployed bytecode (gerçek k=1/matmul/scale=8 devresi) | 14.929 B | Faz C3 |
| setup + prove (gerçek devre, k=1/matmul/scale=8) | 37.3s + 37.4s | Faz C3 |
| verify_gas (gerçek devre, zincir üstü) | 1.174.788 | Faz C3 |
| max_abs_error (gerçek devre, fidelity) | %3,21 (bağıl) | Faz C3 |

**Not:** Faz D'nin kendi test koşumu (`tests/test_contracts.py`) KÜÇÜK
bir sentetik devre kullandığından (hızlı test iterasyonu için — yukarıda
"Test kapsamı" bölümünde açıklandı), kendi gas/süre rakamları GERÇEK
üretim devresini (k=1/matmul/scale=8, `configs/circuit.yaml`) temsil
ETMİYOR — operasyonel maliyet referansı hâlâ Faz C3'ün ölçtüğü
değerlerdir (yukarıdaki tablonun alt üç satırı). İlginç bir gözlem:
sentetik test devresinin verifier'ı (14.921 B) gerçek devrenin
verifier'ına (14.929 B) neredeyse ÖZDEŞ boyutta — bu, Halo2 verifier
bytecode boyutunun küçük ölçekte devre-spesifik değil, Halo2/KZG
doğrulama mantığının SABİT (boilerplate) kısmı tarafından domine
edildiğini düşündürüyor.

## Yol boyunca çözülen dört veri-biçimi tuzağı

Bu dördü, ezkl (Rust/pyo3 tabanlı) ile web3.py (Python/JSON-RPC tabanlı)
arasında köprü kurarken karşılaşılan, GENELLENEBİLİR entegrasyon
tuzaklarıdır — hiçbiri resmi ezkl dokümantasyonunda AÇIKÇA
belgelenmemişti, hepsi gerçek Colab koşumlarında ampirik olarak ortaya
çıktı.

### 1. EIP-1559 / legacy `gasPrice` çakışması

**Belirti:** `TypeError: Unknown kwargs: ['gasPrice']`.
**Kök sebep:** `Contract.constructor(...).build_transaction({})` bağlı
olduğu zincirin (anvil) EIP-1559 desteğini görüp `maxFeePerGas`/
`maxPriorityFeePerGas`'ı KENDİSİ ekliyordu; bizim kodumuz bunun ÜSTÜNE
KOŞULSUZ eski tip `gasPrice` de ekliyordu — `eth-account` ikisini bir
arada görünce reddediyordu.
**Ders:** `build_transaction()`'ın hazırladığı bir işlem sözlüğüne ASLA
elle gas-fiyatı alanı EKLEMEYİN — hangi tipin (EIP-1559/legacy) zaten
orada olduğunu KONTROL EDİN, yoksa BİR TİP seçip ekleyin, asla ikisini
karıştırmayın. Çözüm: `chain/client.py: _build_tx` — tek, test edilmiş
(`tests/test_client.py`) ortak yol.

### 2. `public_inputs`: little-endian, iç içe hex string listesi

**Belirti:** `Argument 4 value [...] is not compatible with type
uint256[]`; ham `int(x, 16)` ile okunursa DEVASA, yanlış sayılar çıkıyor.
**Kök sebep:** ezkl'nin `proof.json:instances` alanı iç içe bir liste
(`[[v1, v2, ...]]`) ve her eleman **LITTLE-ENDIAN** bir hex string
(`"0600...00"` = 6, `"...06"` DEĞİL).
**Ders:** Bir hex-encoded alan elemanının endianness'ını ASLA
VARSAYMAYIN — hem little hem big-endian yorumunu hesaplayıp HANGİSİ
beklenen matematiksel aralıkta (burada: BN254 skalar alanı,
`< 21888242871839275222246405745257275088548364400416034343698204186575808495617`)
kalıyorsa onu kullanın, seçimi LOGLAYIN. Çözüm:
`circuits/ezkl_utils.py: parse_public_inputs`.

### 3. `proof`: hex string değil, int listesi

**Belirti:** `Argument 3 value [16, 209, 217, 7, ...] is not compatible
with type bytes`.
**Kök sebep:** `proof.json:proof` alanı bazı ezkl sürüm/yapılandırma
kombinasyonlarında hex string DEĞİL, düz bir INT LİSTESİ (`[0,255]`
aralığında byte değerleri) olarak seri hâle getiriliyor.
**Ders:** İkili veri taşıyan bir JSON alanının biçimini (hex string mi,
int listesi mi, base64 mü) VARSAYMADAN önce GERÇEK bir örnekle kontrol
edin — ikisi de "aynı bilgiyi" taşıdığından yüzeysel olarak makul
görünür, ama tip dönüşümü tamamen farklıdır. Çözüm:
`circuits/ezkl_utils.py: parse_proof_bytes` — hex (önekli/öneksiz), int
listesi, zaten `bytes` olan durumların HEPSİNİ kapsar.

### 4. `ContractLogicError`, `estimate_gas` aşamasında, gönderim ÖNCESİ

**Belirti:** Testler işlem GÖNDERİLİP `receipt.status==0` ile
başarısız olacağını varsayıp `RuntimeError` yakalıyordu, ama gerçekte
`web3.exceptions.ContractLogicError` çok daha ERKEN, `fn_call.build_transaction(...)`
çağrısının KENDİ İÇİNDE (gas tahmini için yaptığı simülasyon sırasında)
fırlıyordu — işlem HİÇ ZİNCİRE GÖNDERİLMİYORDU.
**Kök sebep:** web3.py, "gas" alanı verilmediğinde `eth_estimateGas`'ı
KENDİSİ çağırır; kontrat bir `require` ile revert ederse bu simülasyon
da revert eder ve `ContractLogicError` olarak yükselir — bu, işlemin
GERÇEKTEN gönderilip madenlenmesinden (receipt.status kontrolü) TAMAMEN
AYRI bir başarısızlık noktasıdır.
**Ders:** Bir akıllı kontrat çağrısını saran kod, hem "gas tahmini
sırasında revert" hem "gönderildi ama status=0" durumlarını AYRI AYRI
ele almalı — sadece receipt'i kontrol etmek yetmez. Çözüm:
`chain/client.py: RoundManagerClient._send` artık `build_transaction(...)`'ı
`try/except ContractLogicError` ile sarıp revert mesajını KORUYARAK
(`str(e)`) aynı önekli bir `RuntimeError`'a çeviriyor — çağıran taraf
(ve testler) TEK bir hata tipine/önekine bakabiliyor, ama kontratın
GERÇEK revert metni de mesajın içinde kalıyor.

## Sıradaki adım

Faz D tamamlandı — protokol k=1 ile, gerçek gas/zaman maliyetleriyle,
gerçek bir Halo2Verifier'a karşı doğrulandı. Sıradaki faz (**Faz E**):
mevcut 15 round'un shard'ları üzerinde, hiç yeni eğitim yapmadan, ispat
takvimini (`orchestrator/schedule.py`'nin `full` vs `sampled` modları)
koşturup maliyet karşılaştırma tablosu üretmek (`scripts/replay_proofs.py`).
