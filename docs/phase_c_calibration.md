# Faz C3 Ara Raporu — Kalibrasyon Teşhisi

Bu rapor `scripts/diagnose_calibration.py`'nin Colab'daki ilk (ve tek)
koşumunun gerçek çıktısına dayanır. `scripts/bench_circuit.py`'nin ilk
benchmark denemesi (k=1, embed_mode=matmul, scale=8) `calibrate_settings`'te
şu hatayla düşmüştü:

```
RuntimeError: Failed to calibrate settings: [Uncategorized] calibration
failed, could not find any suitable parameters given the calibration dataset
```

(tepe RAM: 945 MB, decomposition uyarısı: 0). İki olası sebep vardı:
(a) scale=8 gerçekten dar, (b) çoklu-girdi `input.json` şeması yanlış
(`bench_circuit.py`'de "BİLİNÇLİ VARSAYIM" olarak işaretlenmişti).

## Bulgu 1 — Şema DOĞRU, sorun yöntemdi

`ezkl==23.0.5` (`v23.0.5` etiketi, WebFetch ile GERÇEKTEN çekildi,
versiyon kayması riski yok):

- `src/graph/input.rs`: `pub type DataSource = Vec<Vec<FileSourceInner>>`
  — dış liste = graf girdisi başına bir eleman, iç liste = o girdinin
  düzleştirilmiş değerleri. Bizim `{"input_data": [flat_z, flat_c]}`
  şemamız (`input_names=["z","c"]` sırasıyla) bu tanıma uyuyor.
- `ezkl.pyi`: `calibrate_settings(data, model, settings, target,
  lookup_safety_margin, scales, scale_rebase_multiplier, max_logrows)`
  — `scales: Optional[Sequence[int]]` GERÇEKTEN var.

Colab koşumu bunu doğruladı: ONNX grafiğinin girdi şekli ile
`input.json`'daki uzunluklar UYUŞTU, ve **`calibrate_settings(..., scales=[8])`
başarıyla geçti**:

```
method=scales_kwarg, target=resources, scale=8, max_logrows=None
-> başarılı, gerçekleşen input_scale=8, logrows=19, max_abs_error=0.133
```

Buna karşılık `bench_circuit.py`'nin o zamanki yöntemi —
`PyRunArgs.input_scale`'i `gen_settings` zamanında ayarlamak, sonra
`calibrate_settings` sonrası `settings.json`'u elle yamalamak
(`force_settings_scale`) — **hiçbir ölçekte tutmadı**. Sonuç: şema
sorunu değildi, doğru API çağrısı `scales=[scale]` kwarg'ı imiş.
`bench_circuit.py` bu bulguya göre güncellendi (`run_ezkl_pipeline`
artık `calibrate_settings(..., scales=[scale])` kullanıyor,
`force_settings_scale` kaldırıldı, yerine sadece-okuyan
`read_realized_scale` geldi).

## Bulgu 2 — Beklenenin TERSİ: düşük ölçek çalışıyor, yüksek ölçek düşüyor

| scale | target | durum | hata |
|---|---|---|---|
| 8 | resources | ✅ başarılı | — (`max_abs_error=0.133`, `logrows=19`) |
| 11 | resources | ❌ düştü | `[halo2] General synthesis error` |
| 13 | resources | ❌ düştü | `[tensor] significant bit truncation when instantiating, try lowering the scale` |
| 16 | resources | 💥 panik | `pyo3_runtime.PanicException` (bkz. Bulgu 3) |

Normalde daha yüksek `scale` (fixed-point hassasiyeti) daha iyi
fidelity anlamına gelir ve genelde önce DÜŞÜK ölçeklerde darlık
beklenir. Burada tam tersi gözlendi: **scale=8 çalışıyor, 11/13/16
başarısız/panik**. En olası açıklama: `mapping.fc1` katmanının
512-terimlik iç çarpımları (bkz. `docs/phase_a_report.md` — gerçek
mapping ağının en büyük doğrusal katmanı) yüksek ölçekte toplam
büyüklüğün ezkl'in decomposition/lookup taban aralığını aşmasına yol
açıyor — `docs/phase_b_report.md`'deki oyuncak MLP'nin (32 terimlik
iç çarpım) 13/13 ölçeğinde gördüğü "decomposition error: integer ...
is too large" uyarılarının (o zaman fatal değildi) daha büyük bir
versiyonu, ama burada gerçekten FATAL hale geliyor. Bu, makale için
somut bir mühendislik bulgusu: **devre boyutu (iç çarpım terim sayısı)
büyüdükçe kullanılabilir scale aralığı DARALIYOR**, dolayısıyla
"gerçek" bir devrede toy-model'den öğrenilen ölçek varsayımları
geçerli olmayabiliyor.

`scripts/bench_circuit.py`'nin ölçek ızgarası bu yüzden `{8,11,13}`'ten
`{6,7,8,9,10}`'a değiştirildi — amaç, çalışan sınırın (8) etrafında
ince bir tarama yapıp gerçekte ÇALIŞAN en yüksek scale'i (dolayısıyla
en iyi fidelity/güvenli aralığı) bulmak.

## Bulgu 3 — pyo3 `PanicException`, `except Exception` tarafından yakalanmıyor

scale=16 denemesinde ezkl'nin Rust tarafı gerçek bir Rust `panic!()`
fırlattı:

```
pyo3_runtime.PanicException: assertion `left == right` failed
  left: 160
  right: 144
```

`PanicException` Python'da `BaseException`'dan türer (`Exception`'dan
DEĞİL) — bu yüzden hem `diagnose_calibration.py`'nin hem
`bench_circuit.py`'nin o zamanki `except Exception as e:` blokları bunu
YAKALAMADI, script'i düşürdü (ilk koşumda tarama scale=16'da durdu).

**Düzeltme:** her iki script'te de artık

```python
except (KeyboardInterrupt, SystemExit):
    raise
except BaseException as e:
    status = classify_exception_status(e)  # "panic" (PanicException) ya da "failed"
    ...
```

deseni kullanılıyor (`scripts/bench_circuit.py: classify_exception_status`,
`diagnose_calibration.py`'de yeniden kullanılıyor — DRY). Bir kombinasyon
panikleyince artık script düşmüyor, o kombinasyon `status="panic"` ile
ayrı işaretlenip (sıradan `"failed"`'den ayrı — hangi kombinasyonların
ezkl'i gerçekten ÇÖKERTTİĞÜ, hangilerinin sadece normal bir hata
döndürdüğü karışmasın) diğerlerine devam ediliyor.

## Ölçek/fidelity ödünleşimi — makale tablosu için

scale=8'de `max_abs_error=0.133`. Referans `w`'nin gerçek büyüklüğü
(`abs_max`, C0'ın gerçek StyleGAN-XL çıktısından) ile oranlandığında bu
yaklaşık **%12,5 bağıl hata** anlamına geliyor (w aralığı ~5 civarında).
Bu, düşük scale'in getirdiği somut hassasiyet kaybı — `bench_circuit.py`
artık her kombinasyon için hem `max_abs_error` hem
`max_abs_error_relative_pct`'i (`w_abs_max`'a oranlanmış, worker'ın
kendi referans verisinden hesaplanmış) `bench_results.json`'a ve
`docs/phase_c_bench.md`'ye yazıyor — {6,7,8,9,10} ızgarası tamamlandığında
"scale arttıkça hata azalıyor ama scale 11+ hiç derlenmiyor" ödünleşimi
tek bir tabloda görülebilecek.

## Sıradaki adım

`scripts/bench_circuit.py --env colab --only-first` (artık scale=6 ile
başlıyor) ve ardından tam `{6,7,8,9,10}` ızgarası — hangi scale'in
gerçekten en yüksek ÇALIŞAN değer olduğunu ve o noktadaki tam
kısıt/süre/fidelity profilini belirleyecek.
