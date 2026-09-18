"""Faz F: FID/KID + sınıf-tutarlılığı ölçümü.

**Yeni bir ölçüm yolu İCAT EDİLMEDİ.** FID/KID, StyleGAN-XL'in KENDİ
`metrics.metric_main.calc_metric` fonksiyonu ÇAĞRILARAK hesaplanır
(WebFetch ile `autonomousvision/stylegan-xl`'in gerçek kaynağından
doğrulandı: `calc_metric(metric, **kwargs)` -> kwargs
`metrics.metric_utils.MetricOptions(G, dataset_kwargs, num_gpus, rank,
device, ...)`'a gider). Bu, mevcut deneyin `fid50k_full=13.13` gibi
sayılarıyla karşılaştırılabilir kalmanın TEK yolu — dataset/
çözünürlük/max_size gibi hiçbir ayar burada yeniden yazılmıyor,
`load_metric_options_from_training_options` bunları GERÇEK
`training_options.json`'dan okuyor.

**Sınıf tutarlılığı (class_confusion_matrix)** ise StyleGAN-XL'in
KENDİSİNDE olmayan, bu fazın YENİ ihtiyacı (`conditional_poison`
saldırısının FID'de görünmeyen zararını ölçmek için) — ama YİNE DE
FID/KID ile AYNI özellik uzayını (aynı Inception dedektörü,
`DETECTOR_URL`) ve StyleGAN-XL'in `kernel_inception_distance.compute_kid`
fonksiyonunun BİREBİR aynı (WebFetch ile doğrulanmış) polinom-çekirdek
formülünü kullanıyor — sadece bu formülü rastgele bir gerçek/üretilmiş
çift yerine SINIF-FİLTRELENMİŞ çiftlere uyguluyor (StyleGAN-XL'in
resmi `compute_kid`'i sınıf filtrelemeyi desteklemiyor, bu yüzden
formül `compute_kid_from_features` olarak buraya taşındı).

BİLİNÇLİ TEST SINIRI: gerçek StyleGAN-XL reposu (`metrics/metric_main.py`,
`metrics/metric_utils.py`, `dnnlib`), gerçek bir eğitilmiş `G_ema`
ve gerçek APTOS/DR veri kümesi gerektiren fonksiyonlar (`run_official_metric`,
`get_feature_detector`, `extract_features`, `generate_class_images`,
`collect_real_class_images`, `class_confusion_matrix`) SADECE Colab'da
çalışır/test edilir. Saf/dosya-tabanlı yardımcılar
(`load_metric_options_from_training_options`, `compute_kid_from_features`,
`build_fixed_class_batch`, `to_uint8_images`) `tests/test_metrics.py`'de
yerelde GERÇEKTEN test edilir.
"""

from __future__ import annotations

import inspect
import json
import os
import time

import numpy as np
import torch

# StyleGAN-XL'in `metrics/kernel_inception_distance.py`'sinin kid50k_full
# için kullandığı SABİT Inception dedektörü — WebFetch ile doğrulandı.
# Sınıf-tutarlılığı ölçümünün FID/KID ile AYNI özellik uzayında kalması
# için BİREBİR aynı URL/kwargs kullanılıyor.
DETECTOR_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
DETECTOR_KWARGS = {"return_features": True}

# StyleGAN-XL'in resmi kid50k_full'unun kullandığı sabitler (WebFetch ile
# doğrulandı) — compute_kid_from_features'ın varsayılanları BİREBİR bunlar.
KID_NUM_SUBSETS = 100
KID_MAX_SUBSET_SIZE = 1000


# ---------------------------------------------------------------------------
# Saf/dosya-tabanlı yardımcılar (StyleGAN-XL/GPU GEREKTİRMEZ)
# ---------------------------------------------------------------------------


def load_metric_options_from_training_options(training_options_path: str, dataset_root_override: str | None = None) -> dict:
    """`training_options.json`'dan (checkpoint'in YANINDA duran, GERÇEK
    eğitim koşumunun kendi kaydettiği dosya — `autonomousvision/stylegan-xl:
    train.py`'de `c.training_set_kwargs`/`c.num_gpus`/`c.metrics` olarak
    yazılıyor, WebFetch ile doğrulandı) FID/KID için gereken ÜÇ alanı
    okur. Hiçbir dataset yolu/çözünürlük/max_size burada VARSAYILMAZ —
    hepsi dosyadan okunuyor, bu da "mevcut ölçüm ayarlarıyla birebir
    aynı" isteğinin somutlaşması.

    `dataset_root_override`: kayıtlı `training_set_kwargs["path"]`
    ORİJİNAL eğitim makinesindeki mutlak bir yol olabilir — bu Colab
    oturumunda GEÇERSİZ olabilir. Verilirse, kayıtlı yolun SADECE son
    bileşeni (dosya/klasör adı) `dataset_root_override` altına yeniden
    konumlandırılır; verilmezse dosyadaki yol OLDUĞU GİBİ kullanılır."""
    with open(training_options_path, encoding="utf-8") as f:
        data = json.load(f)

    required = ("training_set_kwargs", "num_gpus", "metrics")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"'{training_options_path}' içinde eksik alanlar: {missing}. "
            f"Bulunan üst-seviye anahtarlar: {sorted(data.keys())}"
        )

    dataset_kwargs = dict(data["training_set_kwargs"])
    if dataset_root_override:
        original_path = dataset_kwargs.get("path")
        if not original_path:
            raise ValueError(f"'{training_options_path}': training_set_kwargs içinde 'path' yok, dataset_root_override uygulanamıyor.")
        new_path = os.path.join(dataset_root_override, os.path.basename(original_path.rstrip("/\\")))
        print(f"[eval.metrics] dataset_kwargs['path'] override: '{original_path}' -> '{new_path}'")
        dataset_kwargs["path"] = new_path

    resolved_path = dataset_kwargs.get("path")
    if resolved_path and not os.path.exists(resolved_path):
        override_note = " + dataset_root_override uygulandı" if dataset_root_override else ""
        raise FileNotFoundError(
            f"Dataset yolu bulunamadı: '{resolved_path}' (training_options.json'dan okundu{override_note}). "
            f"--dataset-root ile doğru kökü göster."
        )

    return {"dataset_kwargs": dataset_kwargs, "num_gpus": data["num_gpus"], "metrics": data["metrics"]}


def compute_kid_from_features(
    gen_features,
    real_features,
    *,
    num_subsets: int = KID_NUM_SUBSETS,
    max_subset_size: int = KID_MAX_SUBSET_SIZE,
    rng: np.random.Generator | None = None,
) -> float:
    """StyleGAN-XL'in `metrics/kernel_inception_distance.py: compute_kid`'inin
    BİREBİR aynı formülü (WebFetch ile gerçek kaynaktan doğrulandı) —
    tek fark: hangi görüntü kümelerinin karşılaştırılacağına ÇAĞIRAN
    TARAF karar veriyor (ör. sınıf-filtrelenmiş bir alt küme);
    StyleGAN-XL'in resmi `compute_kid`'i SADECE tüm veri kümesine karşı
    çalışabiliyor, sınıf filtreleme desteği YOK. Polinom çekirdek
    (derece 3), resmi fonksiyonla SAYISAL OLARAK ÖZDEŞ."""
    if rng is None:
        rng = np.random.default_rng()
    gen_features = np.asarray(gen_features, dtype=np.float64)
    real_features = np.asarray(real_features, dtype=np.float64)
    if gen_features.ndim != 2 or real_features.ndim != 2:
        raise ValueError(
            f"gen_features/real_features 2 boyutlu (N, feature_dim) olmalı, "
            f"alınan şekiller: {gen_features.shape}, {real_features.shape}"
        )
    if gen_features.shape[1] != real_features.shape[1]:
        raise ValueError(f"özellik boyutları uyuşmuyor: {gen_features.shape[1]} vs {real_features.shape[1]}")

    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    if m < 2:
        raise ValueError(
            f"KID için en az 2 örnek gerekli (hesaplanan alt küme boyutu m={m}) — "
            f"gen={gen_features.shape[0]} örnek, real={real_features.shape[0]} örnek."
        )

    t = 0.0
    for _ in range(num_subsets):
        x = gen_features[rng.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[rng.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return float(t / num_subsets / m)


def build_fixed_class_batch(batch_size: int, class_index: int, z_dim: int, c_dim: int, *, generator: torch.Generator) -> tuple:
    """`class_index`'e SABİTLENMİŞ (one-hot) `c` + rastgele `z`'den
    oluşan bir batch üretir — `generate_class_images`'ın saf/testable
    çekirdeği."""
    if not (0 <= class_index < c_dim):
        raise ValueError(f"class_index={class_index}, c_dim={c_dim} aralığının ([0,{c_dim})) DIŞINDA.")
    z = torch.randn(batch_size, z_dim, generator=generator)
    c = torch.zeros(batch_size, c_dim)
    c[:, class_index] = 1.0
    return z, c


def _force_kwarg_wrapper(original_fn, kwarg_name: str, kwarg_value):
    """`original_fn`'i, HER çağrıda `kwarg_name=kwarg_value`'yu
    ZORLAYAN bir sarmalayıcıya çevirir — çağıran taraf o kwarg'ı hiç
    geçmese (StyleGAN-XL'in internal katmanlarının YAPTIĞI gibi) BİLE
    zorlanan değer kullanılır. `force_stylegan_ops_impl`'in saf,
    testable çekirdeği — StyleGAN-XL'e bağımlı DEĞİL, herhangi bir
    fonksiyonla test edilebilir."""

    def wrapped(*args, **kwargs):
        kwargs[kwarg_name] = kwarg_value
        return original_fn(*args, **kwargs)

    return wrapped


def to_uint8_images(raw_images: torch.Tensor) -> torch.Tensor:
    """StyleGAN-XL'in ÖZGÜN ölçekleme formülü (WebFetch ile
    `metrics/metric_utils.py: compute_feature_stats_for_generator`'dan
    doğrulandı): `(img * 127.5 + 128).clamp(0, 255).to(uint8)` —
    generator çıktısının `[-1,1]` aralığından `[0,255]` uint8'e
    dönüşümü. FID/KID'nin resmi hesaplamasıyla BİREBİR aynı olması
    için burada YENİDEN İCAT EDİLMEDİ, kaynaktan alındı."""
    return (raw_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# StyleGAN-XL/GPU/gerçek veri kümesine bağımlı adımlar
# (BİLİNÇLİ TEST SINIRI — sadece Colab'da çalışır/test edilir)
# ---------------------------------------------------------------------------

# StyleGAN-XL'in CUDA-derlemeli üç custom-op modülü — WebFetch ile
# gerçek kaynaktan doğrulandı: `bias_act.bias_act`/`upfirdn2d.upfirdn2d`/
# `filtered_lrelu.filtered_lrelu`'nün ÜÇÜ de `impl='cuda'` VARSAYILAN
# değeriyle çağrılıyor, `training/networks_stylegan3.py` gibi internal
# katman kodu `impl=` kwarg'ını HİÇ GEÇMİYOR. `_ORIGINAL_OPS_FUNCTIONS`
# `force_stylegan_ops_impl` TEKRAR TEKRAR çağrılsa bile (ör. önce
# 'cuda' denenip sonra 'ref'e düşülse) HER ZAMAN gerçek orijinal
# fonksiyona sarmalayabilmek için bir kerelik dolduruluyor.
_OPS_MODULE_ATTRS = {"bias_act": "bias_act", "upfirdn2d": "upfirdn2d", "filtered_lrelu": "filtered_lrelu"}
_ORIGINAL_OPS_FUNCTIONS: dict = {}


def force_stylegan_ops_impl(impl: str) -> None:
    """StyleGAN-XL'in üç CUDA-derlemeli custom-op modülünün
    (`torch_utils.ops.bias_act`/`upfirdn2d`/`filtered_lrelu`) HER
    ÇAĞRISINI `impl` değerine ZORLAR.

    **Neden `_init()`'i yamalamak yerine bu yol seçildi:** WebFetch ile
    gerçek kaynak doğrulandı — bu depodaki `bias_act._init()`
    derleme BAŞARISIZ olsa bile İÇ MANTIK gereği `False` DÖNMÜYOR
    (hata `custom_ops.get_plugin`'den `bias_act()` çağrısına kadar
    YÜKSELİYOR, `ModuleNotFoundError: No module named 'bias_act_plugin'`
    tam BÖYLE ortaya çıkıyor). `_init()`'i yamalamak StyleGAN'ın bu iç
    (ve versiyon-kırılgan) mantığına bağımlı olurdu. Bunun yerine üç
    modülün KENDİ genel işlevi sarmalanıp `impl` HER ÇAĞRIDA ZORLANIYOR
    — hangi katman `impl=` kwarg'ını geçerse geçsin (StyleGAN-XL'in
    kendi katmanları HİÇ geçmiyor) SONUÇ her zaman aynı.

    `impl='ref'`: saf PyTorch referans uygulaması — matematiksel olarak
    AYNI sonucu verir (CUDA derlemesi GEREKMEZ, ama YAVAŞ). `impl='cuda'`:
    orijinal davranışa (derleme başarılıysa hızlı yol) DÖNÜŞ."""
    if impl not in ("ref", "cuda"):
        raise ValueError(f"impl {impl!r} geçersiz — 'ref' ya da 'cuda' olmalı.")

    import torch_utils.ops.bias_act as bias_act_module
    import torch_utils.ops.filtered_lrelu as filtered_lrelu_module
    import torch_utils.ops.upfirdn2d as upfirdn2d_module

    modules = {"bias_act": bias_act_module, "upfirdn2d": upfirdn2d_module, "filtered_lrelu": filtered_lrelu_module}
    for label, attr_name in _OPS_MODULE_ATTRS.items():
        module_obj = modules[label]
        if label not in _ORIGINAL_OPS_FUNCTIONS:
            _ORIGINAL_OPS_FUNCTIONS[label] = getattr(module_obj, attr_name)
        setattr(module_obj, attr_name, _force_kwarg_wrapper(_ORIGINAL_OPS_FUNCTIONS[label], "impl", impl))

    print(f"[eval.metrics] StyleGAN-XL custom op'ları (bias_act/upfirdn2d/filtered_lrelu) impl='{impl}' olarak ZORLANDI.")


def run_official_metric(metric_name: str, g_ema, *, dataset_kwargs: dict, num_gpus: int = 1, device=None) -> dict:
    """StyleGAN-XL'in KENDİ `metrics.metric_main.calc_metric`'ini
    ÇAĞIRIR — FID/KID hesaplama mantığı burada YENİDEN YAZILMIYOR.
    `g_ema` çağıranın (ör. `scripts/run_attacks.py`) zaten yüklediği,
    state_dict'i FedAvg/saldırı sonucuyla DEĞİŞTİRİLMİŞ gerçek bir
    `G_ema` modülü olmalı. StyleGAN-XL reposunun `sys.path`'te olduğu
    (`fl.stylegan_xl_env.ensure_stylegan_xl_on_path`) çağıran tarafça
    ÖNCEDEN sağlanmış olmalı."""
    from metrics import metric_main

    result = metric_main.calc_metric(metric_name, G=g_ema, dataset_kwargs=dataset_kwargs, num_gpus=num_gpus, rank=0, device=device)
    return dict(result.results)


def get_feature_detector(device):
    from metrics import metric_utils

    return metric_utils.get_feature_detector(url=DETECTOR_URL, device=device, num_gpus=1, rank=0, verbose=True)


def extract_features(detector, images_uint8: torch.Tensor, device, batch_size: int = 64) -> np.ndarray:
    """`images_uint8` (N,C,H,W uint8) üzerinde batch batch Inception
    özellik çıkarımı yapar — `metrics/metric_utils.py`'nin dedektör
    çağrı deseniyle (`detector(images, **DETECTOR_KWARGS)`) AYNI."""
    features = []
    for start in range(0, images_uint8.shape[0], batch_size):
        batch = images_uint8[start : start + batch_size].to(device)
        with torch.no_grad():
            f = detector(batch, **DETECTOR_KWARGS)
        features.append(f.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def generate_class_images(
    g_ema, *, class_index: int, c_dim: int, z_dim: int, num_images: int, device, batch_size: int = 32, seed: int = 0
) -> torch.Tensor:
    """`class_index`'e sabitlenmiş `num_images` görüntü üretir.
    `g_ema.forward`'ın gerçek imzası `inspect.signature` ile kontrol
    edilir (Faz C0'daki desenin aynısı) — `noise_mode` parametresi
    VARSA `"const"` geçilir (deterministik üretim için), YOKSA
    varsayılan davranışa güvenilip UYARI basılır (varsayımda
    bulunulmaz)."""
    forward_params = inspect.signature(g_ema.forward).parameters
    extra_kwargs: dict = {}
    if "noise_mode" in forward_params:
        extra_kwargs["noise_mode"] = "const"
    else:
        print("[eval.metrics] UYARI: G_ema.forward'da 'noise_mode' parametresi yok — varsayılan davranış kullanılacak.")

    generator = torch.Generator().manual_seed(seed)
    chunks = []
    remaining = num_images
    with torch.no_grad():
        while remaining > 0:
            b = min(batch_size, remaining)
            z, c = build_fixed_class_batch(b, class_index, z_dim, c_dim, generator=generator)
            img = g_ema(z.to(device), c.to(device), **extra_kwargs)
            chunks.append(to_uint8_images(img).cpu())
            remaining -= b
    return torch.cat(chunks, dim=0)


def collect_real_class_images(dataset_kwargs: dict, class_index: int, num_images: int) -> torch.Tensor:
    """Gerçek veri kümesinden (StyleGAN-XL'in `training.dataset.Dataset`
    uyumlu sınıfı, `dataset_kwargs["class_name"]` üzerinden
    `dnnlib.util.construct_class_by_name` ile kurulur) `class_index`'e
    ait İLK `num_images` görüntüyü toplar. `get_label(idx)`'in
    WebFetch ile doğrulanmış davranışı: int64 etiketler one-hot
    float32'ye çevrilir, `use_labels=False` ise BOŞ dizi döner (bu
    durumda sınıf filtreleme İMKANSIZ, net hata verilir — sessizce
    "hepsi 0. sınıf" varsayılmaz)."""
    import dnnlib

    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    matching_indices: list[int] = []
    for idx in range(len(dataset)):
        label = dataset.get_label(idx)
        if label.size == 0:
            raise RuntimeError(
                f"Veri kümesi etiketsiz (use_labels=False, dataset_kwargs={dataset_kwargs}) — "
                f"sınıf bazlı filtreleme yapılamaz."
            )
        label_class = int(label.argmax()) if label.size > 1 else int(round(float(label.reshape(-1)[0])))
        if label_class == class_index:
            matching_indices.append(idx)
        if len(matching_indices) >= num_images:
            break

    if not matching_indices:
        raise RuntimeError(f"Veri kümesinde sınıf {class_index}'e ait HİÇ görüntü bulunamadı.")
    if len(matching_indices) < num_images:
        print(f"[eval.metrics] UYARI: sınıf {class_index} için istenen {num_images} yerine sadece {len(matching_indices)} gerçek görüntü bulundu.")

    images = np.stack([dataset[i][0] for i in matching_indices])
    return torch.from_numpy(images)


def class_confusion_matrix(
    g_ema,
    *,
    dataset_kwargs: dict,
    num_classes: int,
    z_dim: int,
    device,
    num_images_per_class: int = 200,
    batch_size: int = 32,
    seed: int = 0,
) -> dict:
    """Faz F'nin sınıf-tutarlılığı metriği: `num_classes × num_classes`
    bir KID matrisi — `kid_matrix[g][r]` = "sınıf g için ÜRETİLEN
    görüntüler" ile "sınıf r'ye ait GERÇEK görüntüler" arasındaki KID.
    `perceived_class[g] = argmin_r kid_matrix[g][r]` — üretilen g
    sınıfının GERÇEKTE hangi sınıfa En Yakın (özellik uzayında)
    olduğu. `perceived_class[g] != g` ise bir SINIF KARIŞIKLIĞI/TAKASI
    tespit edilmiş demektir (`conditional_poison` saldırısının asıl
    ölçütü — bkz. modül docstring'i ve `docs/phase_f_attacks.md`)."""
    device_ = device
    detector = get_feature_detector(device_)
    generation_seconds = 0.0

    real_features_by_class: dict[int, np.ndarray] = {}
    for r in range(num_classes):
        real_images = collect_real_class_images(dataset_kwargs, r, num_images_per_class)
        real_features_by_class[r] = extract_features(detector, real_images, device_)
        del real_images

    kid_matrix: list[list[float]] = [[0.0] * num_classes for _ in range(num_classes)]
    perceived_class: list[int] = [0] * num_classes
    for g in range(num_classes):
        t_gen = time.perf_counter()
        gen_images = generate_class_images(
            g_ema, class_index=g, c_dim=num_classes, z_dim=z_dim, num_images=num_images_per_class,
            device=device_, batch_size=batch_size, seed=seed,
        )
        generation_seconds += time.perf_counter() - t_gen
        gen_features = extract_features(detector, gen_images, device_)
        del gen_images
        rng = np.random.default_rng(seed)
        for r in range(num_classes):
            kid_matrix[g][r] = compute_kid_from_features(gen_features, real_features_by_class[r], rng=rng)
        perceived_class[g] = int(np.argmin(kid_matrix[g]))

    swap_detected = [perceived_class[g] != g for g in range(num_classes)]
    return {
        "kid_matrix": kid_matrix,
        "perceived_class": perceived_class,
        "swap_detected": swap_detected,
        "num_images_per_class": num_images_per_class,
        "generation_seconds": generation_seconds,
        "num_generated_images": num_classes * num_images_per_class,
    }
