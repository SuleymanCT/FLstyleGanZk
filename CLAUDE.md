# CLAUDE.md

Bu dosya, bu repoda çalışırken uyulması gereken sabit kuralları listeler.

## Kurallar

1. **Drive'daki korumalı köklerin hepsi SALT OKUNURDUR:**
   `Generative_Image`, `FL_Experiments`, `parallel_final`, `StyleGANTrain`
   (tam liste: `storage/pathguard.py: PROTECTED_MARKERS`). Bu köklerin
   altındaki hiçbir yola hiçbir kod yazmaz, hiçbir şey silmez. Sadece
   okuma (`legacy.load_network_pkl`, `torch.load` vb.) yapılır.

2. **Base model yeniden eğitilmez. 15 round yeniden eğitilmez.** Mevcut
   deney tamamlanmış ve eğitilmiş kabul edilir; bu repo sadece onun
   üzerine bir ispat/doğrulama katmanı ekler.

3. **StyleGAN-XL eğitim döngüsünün matematiğine dokunulmaz.** ZK ve
   zincir katmanı yalnızca round sınırındaki kaydedilmiş ağırlıklar
   (`network-snapshot.pkl`, `fedavg_N.pt`) üzerinde çalışır — eğitim
   kodunun içine girmez, eğitim adımlarını değiştirmez.

4. **Discriminator hiçbir kod yolunda site dizininden çıkmaz.** Sadece
   generator (G_ema / mapping + sınıf gömme) ağırlıkları shard'lanır,
   paylaşılır, ispatlanır.

5. **Kuantizasyon yalnızca ispat kopyasında yapılır.** FedAvg agregasyonu
   her zaman fp32 üzerinden çalışır; ezkl'e giden kopya ayrı bir
   kuantize kopyadır, orijinal fp32 ağırlıkları değiştirmez.

6. **Faz B bittikten sonra mock/dummy/placeholder çıktı YASAKTIR.**
   Gerçek veri/ağırlık/ölçüm yoksa exception fırlatılır — sahte sayı,
   sahte ispat, sahte metrik uydurulmaz. (Faz B'nin kendisi, tanımı
   gereği, tek istisna olan oyuncak demo fazıdır.)

7. **Ağır işler Colab'da çalışır.** Yerelde GPU eğitimi başlatılmaz,
   yerelde `.pkl` dosyası (StyleGAN pickle) okunmaz/açılmaz. Yerel
   ortam sadece CPU işleri (kontratlar, ezkl devreleri, orkestratör,
   testler) için kullanılır.

8. **Her değişiklikten sonra `pytest` çalıştırılır.** Kırık test
   bırakılmaz; bir değişiklik mevcut bir testi bozuyorsa, ya değişiklik
   düzeltilir ya da test bilinçli olarak güncellenir (sessizce
   atlanmaz).

9. **Yazan HER script, her dosya yazma çağrısından hemen önce
   `storage.pathguard.assert_writable(path)` çağırır.** `torch.save`,
   `open(path, "w")`, `os.makedirs`, `shutil.*` — hepsinden önce. Madde
   1'deki kural artık sadece dokümantasyona değil, buna dayanır:
   korumalı bir yola yazma girişimi programatik olarak `RuntimeError`
   ile durmalı. Korumalı köklerden okurken de (`network-snapshot.pkl`,
   `fedavg_*.pt` gibi büyük ikili dosyalar) `storage.pathguard.open_readonly`
   kullanılır.
