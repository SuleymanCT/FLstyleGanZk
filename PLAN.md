# PLAN.md

Altı fazın tanımı ve kabul kriterleri. Her fazın durumu, o faz
tamamlandığında güncellenir.

## Faz A (Colab) — Envanter ve çıkarma

**durum: yapılmadı**
(kod yazıldı — bkz. `scripts/inventory.py`, `scripts/extract_shards.py`,
`scripts/audit_fedavg.py`, `fl/module_tree.py`, `fl/shard_utils.py`,
`fl/fedavg_utils.py`, `fl/inventory_utils.py`, `fl/stylegan_xl_env.py`;
saf mantık birim testleriyle doğrulandı, ama gerçek Drive verisi
üzerinde Colab'da HENÜZ koşulmadı — kabul kriteri ancak o koşumdan
sonra karşılanabilir.)

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
