# Faz B Raporu — Oyuncak Uçtan Uca ZK İspat Zinciri

Bu rapor Colab'daki gerçek 4. koşumun (Faz B'nin ilk BAŞARILI uçtan uca
koşumu) çıktısına dayanır. Önceki üç koşumun her biri farklı bir gerçek
hatayı ortaya çıkardı; hepsi burada, gerçek sırasıyla belgeleniyor —
hiçbiri uydurulmadı, hepsi gerçekten Colab'da yaşandı.

## Sonuç: geçti

Küçük MLP (32→32→8) → ONNX → ezkl (setup/prove/verify) → Solidity
verifier → solc derleme → anvil'e deploy → web3 ile zincir üstü
doğrulama — zincirin TAMAMI gerçekten çalıştı, ispat gerçekten zincire
gönderildi ve doğrulandı (`verified: True`).

## Adım süreleri

| Adım | Süre |
|---|---:|
| export_onnx | 0.178s |
| ezkl_setup (gen_settings→calibrate→compile→get_srs→setup) | 11.173s |
| ezkl_prove (gen_witness→prove) | 13.566s |
| ezkl_verify_offchain | 0.042s |
| generate_solidity_verifier | 0.191s |
| compile_verifier_solidity | 20.515s |
| deploy_and_verify_onchain | 0.511s |
| **Toplam** | **~46.2s** |

`compile_verifier_solidity` en yavaş adım (20.5s) — 4 farklı (viaIR,
optimizer_runs) kombinasyonunun sırayla denenmesinin maliyeti (bkz.
aşağıdaki "Hangi strateji çalıştı" bölümü); tek bir doğru kombinasyon
bilinseydi bu adım çok daha hızlı olurdu (bu yüzden madde 2'de
`DEFAULT_STRATEGIES` sırası, çalışan kombinasyon ilk sıraya gelecek
şekilde güncellendi).

## Gas ve bytecode

| Metrik | Değer |
|---|---:|
| deploy_gas | 2.956.287 |
| verify_gas | 546.938 |
| deployed_bytecode_size | 13.426 byte |
| EIP-170 sınırı | 24.576 byte |
| Sınıra oran | %54,6 (rahat sınırlar içinde) |

Oyuncak MLP için deploy maliyeti (~3M gas) StyleGAN mapping ağı gibi
çok daha büyük bir devre için ORANTILI OLARAK artacaktır — bu Faz D'nin
`RoundManager.sol` + gerçek verifier deploy maliyeti tahmininde referans
alınmalı.

## Hangi derleme stratejisi çalıştı, hangileri düştü, neden

`chain.solc.compile_with_fallback_strategies`'in bu koşumdaki denemeleri
(solc sürümü sabit `0.8.20` kaldı, hiçbir sürüm değişikliği gerekmedi):

| Sıra | viaIR | optimizer_runs | Sonuç |
|---:|---|---:|---|
| 1 | `True` | 1 | ❌ başarısız |
| 2 | `True` | 50 | ❌ başarısız |
| 3 | `True` | 200 | ❌ başarısız |
| 4 | `False` | 200 | ✅ **BAŞARILI** |

**Bu, üç Colab koşumunun (2., 3., 4. tur) topladığı gerçek hata
zincirinin sonucu:**

1. **(2. tur)** Ad-hoc `solc --combined-json bin` (optimizer KAPALI,
   via-ir yok, solc varsayılanı) → `Error: Stack too deep. Try
   compiling with --via-ir`.
2. **(3. tur)** `viaIR=True, optimizer_runs=200` (hata mesajının
   önerdiği tam olarak buydu) → **farklı** bir hata:
   `YulException: Cannot swap Variable usr$l_blind with Slot
   TMP[mulmod, 0]: too deep in the stack by 1 slots`. Yani via-IR
   önceki hatayı çözmedi, kendi Yul-seviyesi optimizer'ında YENİ bir
   stack-derinliği sorunu çıkardı.
3. **(4. tur, bu koşum)** `viaIR=True` ile `runs=1` ve `runs=50` da
   denendi, ikisi de aynı sınıfta hatalarla düştü (viaIR açıkken sorun
   optimizer agresifliğinden değil, via-IR'in KENDİSİNİN bu yoğun elle
   yazılmış assembly'yi işleme şeklinden kaynaklanıyor gibi görünüyor).
   Son olarak **`viaIR=False, optimizer_runs=200`** (hata mesajının
   önerdiğinin TAM TERSİ — via-IR'i kapatıp klasik codegen'e dönmek)
   ÇALIŞTI.

**Yorum:** ezkl'nin ürettiği Halo2Verifier.sol, elle yazılmış/optimize
edilmiş yoğun `assembly` blokları içeriyor ("No memoryguard was
present" uyarısı da bunu doğruluyor — bu bloklar `memory-safe` olarak
işaretlenmemiş). solc'un via-IR pipeline'ı (kendi Yul optimizer'ı,
kendi stack tahsis stratejisiyle) böyle elle optimize edilmiş
assembly'yi ANALİZ ETMEYE çalışırken KENDİ stack sınırını aşabiliyor;
klasik (legacy) codegen + basit peephole optimizer bu tür kod için
daha öngörülebilir/güvenli kalıyor. **Genel ders:** "Stack too deep"
hatasının önerdiği `--via-ir` çözümü HER ZAMAN doğru değil — özellikle
zaten yoğun assembly içeren, derleyicinin optimize etmesi değil
OLDUĞU GİBİ geçirmesi beklenen kontratlarda tam tersi (via-IR'i
KAPATIP klasik optimizer'ı açmak) gerekebilir. `compile_with_fallback_strategies`
bu yüzden varsayımda bulunmak yerine gerçekten dener.

## Decomposition uyarılarının durumu

Faz B'nin 1. koşumunda (`ezkl_setup`/`ezkl_prove` sırasında)
"decomposition error: integer ... is too large" tipi uyarılar
görülmüştü (Numerical Fidelity Report: `max_abs_error=0.00033`). Bu
koşumda da aynı adımlar (`ezkl_setup` 11.173s, `ezkl_prove` 13.566s)
çalıştı ve **zincir üstünde gerçekten doğrulandı** (`ezkl_verify_offchain`
+ zincir üstü `verify_gas` ile iki kez, hem off-chain hem on-chain) —
yani bu uyarılar SONUÇTA fonksiyonel bir soruna yol açmadı, ispat
geçerliydi ve zincirde de kabul edildi. Bu, PLAN.md Faz C notundaki
değerlendirmeyi DOĞRULUYOR: uyarılar zararsız/bilgilendirici, ama gerçek
mapping ağının çok daha büyük iç çarpımlarında (512 terim vs 32) aynı
scale'de (13/13) daha belirgin hale gelip gelmeyeceği hâlâ ayrı bir
soru — C3'ün (k, bit) taramasında izlenmeye devam etmeli.

## Faz C için çıkarım: bytecode boyutu izlenmesi gereken bir metrik

Oyuncak MLP'nin verifier'ı 13.426 byte (EIP-170'in %54,6'sı). Gerçek
mapping devresi (669.248 parametre — oyuncağın ~627 katı büyüklüğünde,
bkz. `docs/phase_a_report.md` Bölüm A) çok daha fazla kısıt/gate
içerecek, bu da verifier kontratının ÇOK daha büyük olmasına yol
açabilir. Halo2/ezkl verifier boyutu genelde kısıt sayısıyla ORANTILI
büyür (yaklaşık doğrusal-yakın) — kaba bir ekstrapolasyonla bile 24.576
byte sınırının aşılması ihtimal dışı değil. Bu yüzden **Faz C'nin C3
benchmark ızgarasına (k×bit taraması) `deployed_bytecode_size` ve
`exceeds_eip170` de birer sütun olarak eklenmeli** — bir (k, bit)
kombinasyonu ispat/doğrulama açısından "başarılı" olsa bile, verifier'ı
deploy edilemiyorsa (EIP-170'i aşıyorsa) o kombinasyon da
"başarısız" sayılmalı (PLAN.md'ye bu not eklendi).
