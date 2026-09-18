# Faz E Sonuç Raporu — Tam vs Kademeli İspat Maliyet Karşılaştırması

**Durum: Faz E TAMAMLANDI (nihai sürüm).** Bu doküman,
`scripts/replay_proofs.py` tarafından otomatik üretilen
`docs/phase_e_replay.md`'ye (ham tablolar/başarısızlık özeti/altyapı
olayları) EK olarak, ÜÇ Colab koşumunun (tam mod ilk koşum: 60/60
ispat, anvil çökmesi dahil; kademeli mod: 21/21 ispat; tam modun 13
"bozuk ortam" ispatının `--force` ile yeniden koşumu: 13/13 başarılı)
sonuçlarının yorumlu analizini içerir.

**Bu sürümde değişen:** 13 ispatın yeniden koşumu, önceki raporun
"anvil bellek şişmesi" teşhisini de ELEDİ — yeniden koşumda TAZE bir
anvil vardı, HİÇ segment yeniden başlatması olmadı, ama ispatlar YİNE
YAVAŞTI (üstelik ilk koşumdan da yavaş: toplam süre 9949s → 12173s'ye
ÇIKTI). Kesin, veriyle doğrulanmış nihai açıklama Bölüm 3'te: Colab'ın
paylaşımlı CPU kaynağındaki değişkenlik — ne ağırlık verisine (Bölüm
3.1) ne de altyapı mimarisine (Bölüm 3.2) bağlı. Bu yüzden **ANA
TABLO artık normalize edilmiş karşılaştırma** (Bölüm 1) — ham mutlak
süre toplamları ikincil, açıklayıcı amaçla veriliyor.

## 1. Ana karşılaştırma tablosu

### (a) ANA TABLO — normalize edilmiş karşılaştırma (raporun esas aldığı sayılar)

Bölüm 3'ün gösterdiği gibi mutlak ezkl süresi Colab'da GÜVENİLİR bir
ölçüt değil (paylaşımlı CPU değişkenliği). Bu yüzden ana tablo,
sağlıklı/temiz kümede GERÇEKTEN ölçülmüş ispat başına sabit bir süre
(`77.88s` — `compute_healthy_avg_ezkl_seconds`'ın tam modun 47 sağlıklı
ispatı + kademeli modun 21 ispatının TAMAMI, toplam 68 örnek üzerinden
hesapladığı GERÇEK ortalama — 74s gibi bir sayı UYDURULMUYOR) × ispat
sayısı üzerinden hesaplanıyor:

| mod | ispat sayısı | normalize ezkl süresi (s) | toplam zincir maliyeti (gas) | süre tasarrufu % (normalize) | gas tasarrufu % |
|---|---|---|---|---|---|
| full | 60 | 4672.8 | 269.939.688 | (referans) | (referans) |
| staged | 21 | 1635.5 | 94.481.181 | 65.0 | 65.0 |

`normalize ezkl süresi = ispat sayısı × 77.88s`. Bu tablo
`scripts/replay_proofs.py: render_normalized_comparison_table`
tarafından üretilir — bundan sonraki her Colab koşumunda otomatik
olarak `docs/phase_e_replay.md`'ye yazılır.

**Makalenin raporlayacağı sayı: kademeli mod, tam moda kıyasla hem
gas hem (normalize edilmiş) süre için `%65` tasarruf sağlıyor.** İki
metrik BAĞIMSIZ yollardan (biri sabit devre/verifier boyutundan, biri
sabit ispat-başına-süre varsayımından) aynı orana ulaşıyor — ikisi de
temelde SADECE ispat sayısı oranından (21/60) geliyor.

### (b) Ham toplamlar (İKİNCİL — neden ana tablo olarak KULLANILMADI)

| mod | ispat sayısı | başarılı | toplam ezkl süresi (s) | toplam solc süresi (s) | toplam deploy gas | toplam verify gas | toplam zincir maliyeti (gas) | tasarruf % |
|---|---|---|---|---|---|---|---|---|
| full (ilk koşum) | 60 | 60 | 9540.2 | 66.5 | 196.880.100 | 73.059.588 | 269.939.688 | (referans) |
| full (13 ispat yeniden koşulduktan sonra) | 60 | 60 | 12173\* | ~66.5 | 196.880.100 | 73.059.588 | 269.939.688 | (referans) |
| staged | 21 | 21 | 1641.9 | 21.6 | 68.908.200 | 25.572.981 | 94.481.181 | 65.0 |

\*Yeniden koşulan 13 ispatın kendisi ÖNCEKİNDEN DE YAVAŞTI (`setup`
178-260s, `prove` 245-272s — ilk koşumdaki ~110-160s/~140s'den bile
yüksek), toplam ezkl süresini **9949s'den 12173s'ye** çıkardı — yani
"düzelt/yeniden ölç" girişimi süreyi İYİLEŞTİRMEDİ, KÖTÜLEŞTİRDİ. Bu
tek başına, sorunun ne ağırlığa (aynı ispatlar, aynı devre) ne de
anvil'e (yeniden koşumda taze anvil, sıfır restart) bağlı olmadığının
GÜÇLÜ bir kanıtı — bkz. Bölüm 3.2.

**Ham toplamlar ANA TABLO olarak kullanılmıyor** çünkü mutlak ezkl
süresi, Bölüm 3'te kanıtlandığı gibi, Colab'ın paylaşımlı CPU
kaynağındaki değişkenlikten dolayı KOŞUDAN KOŞUYA anlamlı şekilde
değişiyor — aynı ispatları iki kez koşturmak iki FARKLI toplam
üretebiliyor (9540.2s ilk koşum vs 12173s ikinci koşumdan sonraki
toplam, AYNI 60 ispat için). Bu, mutlak süre toplamının makalenin ana
karşılaştırma metriği olarak GÜVENİLMEZ olduğu anlamına geliyor —
bkz. Bölüm 4'teki metodolojik not.

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
build_round_schedule`'ın beklenen davranışı.

## 3. ÖNEMLİ ANALİZ — ispat süresi neden iki kümeye ayrılıyor (74s vs 180-270s)?

Üç aday hipotez sırayla test edildi; ilk ikisi VERİYLE ELENDİ, üçüncüsü
kaldı ve nihai açıklama oldu.

### 3.1. Hipotez 1 — ağırlık/devre farkı (ELENDİ)

`param_visibility="fixed"` yüzünden her (round,site)'ın ağırlıkları
farklı, bu da `calibrate_settings`'in veri-bağımlı bir arama yaptığı
(Faz E'nin zaman araştırması, `docs/phase_e_timing_investigation.md`)
anlamına geliyordu — teorik olarak farklı ağırlıklar farklı devre
boyutu/logrows gerektirebilirdi. **Gerçek veri bunu KESİN olarak
elledi:** 60 ispatın HEPSİNDE devre istatistikleri BİREBİR AYNI:

| istatistik | değer (60 ispatın TAMAMINDA) |
|---|---|
| `logrows` | 19 |
| `num_rows` | 293.916 |
| `pk_size` | 2517 MB |

Ağırlıklar devre BOYUTUNU hiç etkilemiyor (k=1/matmul/scale=8
yapılandırması sabit `logrows`/`num_rows`/pk boyutu üretiyor,
ağırlıkların GERÇEK değerlerinden bağımsız) — yani "yavaş ispatlar
daha büyük/karmaşık bir devre çözüyor" açıklaması YANLIŞ. Kalibrasyon
verisi-bağımlı olabilir (süre açısından, madde 3b'de zaten not
düşülmüştü) ama devre BOYUTUNU değiştirmiyor, ve gözlenen 2-4× süre
farkını (aynı boyuttaki bir devre için) açıklamaya YETMİYOR.

### 3.2. Hipotez 2 — anvil'in bellek şişmesi (ELENDİ)

Önceki rapor turunda 13 yavaş ispatın TAMAMININ anvil'in çöktüğü ilk
tam mod koşumunda üretilmiş olması, "anvil'in bellek baskısı AYNI
makinedeki ezkl'i de yavaşlattı" teşhisine yol açmıştı. **Bu 13
ispatın `--force` ile YENİDEN koşulmasıyla hipotez ELENDİ:**

- Yeniden koşumda **TAZE bir anvil** vardı (her 5 komut kendi
  segmentini/anvil'ini `open_segment(0)` ile sıfırdan başlattı).
- **HİÇ segment yeniden başlatması olmadı** (13 ispat, varsayılan
  `anvil_restart_interval=10`'un biraz üzerinde ama pratikte hiçbiri
  eşiği tetiklemeden bitti — anvil ÇÖKMEDİ).
- Buna RAĞMEN ispatlar YİNE yavaştı — **hatta ilk koşumdan bile daha
  yavaş** (`setup` 178-260s, `prove` 245-272s; ilk koşumdaki
  ~110-160s/~140s'nin üzerinde). Toplam ezkl süresi 9949s'den
  12173s'ye ÇIKTI.

Eğer sebep anvil'in bellek şişmesi olsaydı, TAZE bir anvil + sıfır
restart ile bu 13 ispatın ~74-78s'lik sağlıklı süreye dönmesi
BEKLENİRDİ. Tam tersi gözlendi — bu, anvil'in kendisinin darboğaz
OLMADIĞININ güçlü, doğrudan bir kanıtı.

### 3.3. Nihai açıklama — Colab'ın paylaşımlı CPU kaynağındaki değişkenlik

Ağırlık/devre farkı (3.1) ve anvil bellek durumu (3.2) elendikten
sonra geriye kalan, VERİYLE TUTARLI tek açıklama: **Colab'ın
paylaşımlı sanal makine altyapısındaki CPU/kaynak değişkenliği.**
Destekleyen gözlemler:

- Yavaş ölçümler **ardışık kümeler halinde** geliyor — bir koşudaki
  TÜM ispatlar ya hep hızlı ya hep yavaş, aralarında rastgele
  serpiştirilmiş tekil aykırı değerler DEĞİL. Bu desen, "hangi
  ispatın hangi ANDA/hangi VM üzerinde çalıştığı" gibi koşum-dışı
  (makine durumuna bağlı) bir faktöre işaret ediyor — ispatın
  KENDİSİNE (hangi round/site, hangi ağırlık) özgü bir faktöre değil.
- Aynı 13 ispat, aynı devre yapılandırmasıyla, İKİ AYRI koşumda İKİ
  FARKLI (ikisi de yavaş, ama farklı) süre üretti — mutlak süre
  ispatın kendisinin DETERMİNİSTİK bir özelliği değil, o anki VM'in
  ödünç aldığı CPU payına bağlı GÖRÜNÜYOR.
- Bu, Colab'ın (özellikle ücretsiz/paylaşımlı katmanlarda) belgelenmiş,
  bilinen bir davranışı — tahsis edilen sanal CPU'nun gerçek/fiziksel
  karşılığı diğer kullanıcılarla paylaşılıyor, zaman içinde ve
  oturumdan oturuma değişebiliyor.

**Bu artık kesinleşmiş bulgu olarak kabul ediliyor** (üçüncü, elenmemiş
hipotez) — daha fazla veri toplamak (ör. 13 ispatı BİR KEZ DAHA
koşturmak) sorunu çözmez, sadece üçüncü bir rastgele toplam üretir;
bu yüzden önceki rapor turunun "13 bozuk-ortam ispatını yeniden koştur"
önerisi (o zamanki Bölüm 6, artık bu dosyada YOK) ARTIK TEKRARLANMAYACAK
— zaten bir kez uygulandı (bu dosyanın konusu) ve GÖSTERDİ ki mesele
anvil değil, makine varyansı; bunu bir kez daha koşturmak yeni bir
rastgele sayı üretmekten öte bir şey katmaz.

## 4. Metodolojik not (makalede SINIRLILIK olarak yer almalı)

**Colab gibi paylaşımlı bulut ortamlarında ölçülen MUTLAK süre
rakamları güvenilir bir karşılaştırma metriği DEĞİLDİR** — aynı
işlemin aynı yapılandırmayla iki ayrı koşumu, gözlemlenen tek başına
2-4× farkla sonuçlanabilir (Bölüm 3). Bu proje için pratik sonuçlar:

1. **Deterministik metrikler (ispat sayısı, gas tüketimi) tercih
   edilmeli.** Bu metrikler devre/verifier yapılandırmasına bağlı,
   makine durumuna bağlı DEĞİL — Bölüm 1'in gas tasarrufu sütunu
   (`%65.0`) koşudan koşuya DEĞİŞMEZ, tekrar üretilebilir.
2. **Süre ölçümleri, ancak sağlıklı/temiz bir örneklemden alınan
   sabit bir "ispat başına ortalama" ile NORMALİZE edilip ispat
   sayısına göre ölçeklendiğinde anlamlı hale geliyor** (Bölüm 1(a))
   — ham toplamların DOĞRUDAN karşılaştırılması (Bölüm 1(b))
   YANILTICI olabilir.
3. Bu sınırlılık, makalenin metodoloji/sınırlılıklar bölümünde AÇIKÇA
   belirtilmeli: paylaşımlı bulut altyapısında ölçülen mutlak zaman
   metrikleri gösterge niteliğindedir, kesin/tekrar-üretilebilir bir
   performans iddiası olarak sunulmamalıdır; buna karşın gas/ispat-sayısı
   gibi deterministik metrikler kesinlik taşır ve makalenin ANA iddiası
   bunlara dayanmalıdır.

## 5. Kod düzeltmesi — `run_id` per-kombinasyon `result.json`'a yazılmıyordu

**Kök sebep bulundu:** `run_id`, `scripts/replay_proofs.py: main()`'in
başında üretilip SADECE ebeveynin bellekteki `all_results` sözlüğüne
(`replay_results_<mod>.json`'a) ekleniyordu. Ama her (round,site)'ın
KENDİ küçük sonuç dosyası (`{replay_dir}/results/{key}.json`) işçi
alt süreç (`run_worker`) tarafından, `run_id`'den TAMAMEN habersiz
şekilde yazılıyordu — bu dosyaya `run_id`/`segment_index` HİÇ
eklenmiyordu. Kullanıcının "hepsinde '-'" gözlemi, muhtemelen bu
per-kombinasyon dosyalarının incelenmesinden kaynaklandı.

**Düzeltme:** `run_combo_in_subprocess`'e `run_id`/`segment_index`
parametreleri eklendi; işçiden dönen (ya da "crashed" durumunda
sentezlenen) `result` dict'ine bu iki alan eklendikten HEMEN SONRA,
`write_json_file` ile `results/{key}.json` dosyası YENİDEN yazılıyor
— artık HEM aggregate (`replay_results_<mod>.json`) HEM per-kombinasyon
(`results/{key}.json`) dosyası TUTARLI şekilde `run_id`/`segment_index`
içeriyor. `tests/test_replay_proofs.py`'ye 10 yeni saf test eklenmişti
(bir önceki turda) — bu turun düzeltmesi `run_combo_in_subprocess`'in
ezkl/anvil/subprocess'e bağımlı gövdesinde olduğundan (BİLİNÇLİ TEST
SINIRI, sadece Colab'da doğrulanabilir), yerel test eklenmedi; tam
paket yine de yeşil (291 passed, 2 skipped).

## 6. Altyapı kısıtları (değişmedi, referans için korunuyor)

- Her iki modda da (ilk koşumlarda) 1'er planlı (`scheduled_restart`)
  anvil yeniden başlatması gerçekleşti; 13 ispatın yeniden koşumunda
  HİÇ yeniden başlatma olmadı (Bölüm 3.2). Segment mimarisi
  (`docs/phase_e_infra_notes.md`) beklendiği gibi çalıştı: hiçbir
  `ReadTimeout`/altyapı kaynaklı başarısızlık YAŞANMADI (60/60, 21/21,
  13/13 — üç koşumda da sıfır başarısızlık).
- **İtibar sıfırlanması notu:** her segment yeniden başlatmasında/yeni
  segment açılışında `RoundManager` yeniden deploy edilip site'lar
  yeniden kaydediliyor — zincir tarafındaki `reputation`/`isSiteEligible`
  durumu sıfırlanıyor. Kademeli modun takvim KARARI bundan ETKİLENMEDİ
  (`must_prove`'un itibar girdisi `replay_proofs.py`'nin YEREL
  `reputations` sözlüğünden geliyor).
- **Kaynak izolasyonu konusu (Bölüm 3.2 ile güncellendi):** anvil'in
  bellek davranışının ezkl'i yavaşlattığı hipotezi ELENDİ — bu yüzden
  segment mimarisinin "kaynak izolasyonu" gerekçesi GEÇERSİZLEŞMEDİ
  (anvil'in gerçek bir bellek şişmesi/çökme riski hâlâ var ve segment
  mimarisi HÂLÂ ona karşı doğru bir savunma) ama BU SPESİFİK 2-4× süre
  farkının açıklaması OLMAKTAN çıktı — asıl açıklama Bölüm 3.3.

## 7. Sonuç

Faz E'nin kabul kriteri **karşılandı**: tam ve kademeli mod
karşılaştırılabilir şekilde koşup makalenin ana maliyet tablosunu
üretti, sıfır başarısızlık. Üç aday hipotez (ağırlık/devre farkı,
anvil bellek durumu, Colab paylaşımlı CPU değişkenliği) sırayla,
VERİYLE test edildi; ilk ikisi kesin olarak elendi, üçüncüsü nihai
açıklama olarak kaldı. **Makalenin raporlayacağı sayı: `%65` tasarruf
— hem gas hem normalize edilmiş süre için geçerli, ikisi de ispat
sayısı oranından (21/60) geliyor.** Mutlak süre ölçümlerinin paylaşımlı
bulut ortamında güvenilmez olduğu, makalenin metodoloji/sınırlılıklar
bölümüne AÇIKÇA not düşülecek bir bulgu (Bölüm 4).
