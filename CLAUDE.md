# CLAUDE.md

Bu dosya, bu repoda çalışırken uyulması gereken sabit kuralları listeler.

## Kurallar

1. **Drive'daki `FL_Experiments/parallel_final` SALT OKUNURDUR.** Hiçbir
   kod oraya yazmaz, hiçbir kod oradan bir şey silmez. Sadece okuma
   (`legacy.load_network_pkl` vb.) yapılır.

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
