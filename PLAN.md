# PLAN.md

Altı fazın tanımı ve kabul kriterleri. Her fazın durumu, o faz
tamamlandığında güncellenir.

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

## Faz B (yerel) — Oyuncak uçtan uca

**durum: yapılmadı**

Bu projedeki TEK demo fazı. Küçük MLP (32->32->8) -> ONNX -> ezkl
derleme+kalibrasyon -> ispat -> Solidity verifier üretimi -> anvil'de
deploy -> Python'dan doğrulama. `tests/test_toy_pipeline.py` tek
komutla yeşil geçsin, adım süreleri ve gas yazsın. anvil'i test
içinden başlat ve kapat.

**Kabul:** test yeşil, boru hattı çalışıyor. Buradan sonra sahte çıktı
yok.

## Faz C (yerel + Colab) — Gerçek mapping devresi

**durum: yapılmadı**

- **C1** `circuits/rebuild_mapping.py` — shard'dan, StyleGAN-XL'e
  bağımlı OLMAYAN minimal bir torch modülü kur (sadece Linear,
  aktivasyon, normalizasyon). Yapıyı `inventory.json`'daki modül
  ağacından türet. Colab'da üretilecek referans çıktılarla 1e-5 içinde
  eşleşmeli.
- **C2** `circuits/export_mapping.py` — ONNX ihracı. Girdi z (k,512),
  c (k,5) one-hot. onnxruntime ve torch farkı < 1e-4, değilse hata.
- **C3** `circuits/calibrate.py` + `scripts/bench_circuit.py` — ızgara:
  k ∈ {1,4,8} × bit ∈ {8,16}. Her kombinasyon için kısıt sayısı,
  derleme/setup/ispat/doğrulama süresi, tepe RAM, ispat boyutu.
  Patlayan kombinasyon "başarısız" işaretlenir, koşu düşmez.
- **C4** Kuantizasyon etkisi: w_q ve fp32 w arasında kosinüs benzerliği
  ve L2 farkı, DR sınıfı başına ayrı.

**Kabul:** en az bir (k, bit) kombinasyonu 30 dk altında ispat
üretiyor.

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
