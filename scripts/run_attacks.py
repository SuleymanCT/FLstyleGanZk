#!/usr/bin/env python
"""Faz F: saldırı × koruma matrisi — ZK doğrulamasının/norm kontrolünün
zehirleme saldırılarını GERÇEKTEN yakalayıp yakalamadığını FID/KID/
sınıf-tutarlılığı ile ölçer.

**Hiçbir model eğitilmez/yeniden eğitilmez** (CLAUDE.md madde 2/3/6) —
üç saldırı da (`attacks/random_weights.py`, `attacks/scaled_poison.py`,
`attacks/conditional_poison.py`) SADECE ağırlık seviyesinde çalışır.
Bu script Faz A'da ZATEN eğitilmiş, `raw_root`'ta (salt okunur) duran
GERÇEK site checkpoint'lerini (`fl.round_replay.load_site_update`)
alıp bozup FedAvg yapar.

## Akış (her saldırı × her koşul için)

1. Belirtilen `--round`'un 4 site'ının GERÇEK `G_ema.state_dict()`'i
   yüklenir (`fl.round_replay.load_site_update` — Faz D'de zaten
   kanıtlanmış, yeniden yazılmaz).
2. `--round - 1`'in GERÇEK fedavg dosyası (`scripts.audit_fedavg.load_fedavg_file`)
   önceki global olarak yüklenir (`scaled_poison`'ın ΔG hesabı VE
   norm kontrolünün referans noktası — Faz A'nın KENDİ dosyası,
   uydurulmaz).
3. `--poisoned-site` indeksli site'ın state_dict'i seçilen saldırıyla
   bozulur.
4. **UNPROTECTED**: 4 site (3 dürüst + 1 zehirli) `fl.fedavg_utils.RunningAverage`
   ile DOĞRUDAN ortalanır (hiçbir kapı yok) — `orchestrator/aggregate.py`
   YENİDEN YAZILMAZ, sadece kapısız hali (`RunningAverage`) kullanılır.
5. **PROTECTED**: `orchestrator.aggregate.aggregate_round` DOĞRUDAN
   çağrılır — bu fonksiyon zaten "zincir onayı (ZK) + norm kontrolü"
   kapılarının İKİSİNİ de uyguluyor, burada YENİDEN YAZILMIYOR.
   `chain_approved` HER site için `True` verilir (bkz.
   `attacks/detection.py`'nin gerekçesi: ZK bir İÇERİK filtresi
   DEĞİL, taahhüt-ispat-katkı TUTARLILIK kanıtı — bu simülasyondaki
   "dürüst ama bozulmuş" istemci modelinde ZK HER ZAMAN geçerdi).
   Norm eşiğini AŞMAYAN bir saldırı (gerçek Colab bulgusu:
   `conditional_poison`, `||ΔG||=2507 < tau=3000`) protected koşulda
   da DIŞLANMAZ — bu durumda unprotected/protected SONUÇLARI BİREBİR
   AYNI çıkar, bu bir HATA değil, korumanın o saldırıyı DURDURAMADIĞININ
   doğrudan kanıtıdır.
6. **POISONED_ALONE** (H1 tanı koşulu — bkz. `build_poisoned_alone_global`):
   FedAvg'ı TAMAMEN ATLAYIP zehirli site'ın KENDİ ağırlığını "global"
   olarak kullanır — 3 dürüst site'ın seyreltme etkisi OLMADAN saldırı
   TEK BAŞINA nasıl görünüyor, bunu izole eder.
7. Her koşul için de sonuç global `G_ema`'nın ("shell" — 4 site'tan
   birinin GERÇEK, sadece state_dict'i DEĞİŞTİRİLECEK modülü) içine
   yüklenir, `eval/metrics.py` ile `fid50k_full`/`kid50k_full`
   (`--metrics-mode full`) + sınıf-tutarlılığı matrisi hesaplanır.
8. Tespit analizi: `attacks/detection.py: verify_commitment_consistency`
   (ZK) + `orchestrator.aggregate.gate_site_update`'in `delta_norm`/
   dışlama sonucu (norm) — İKİSİ de GERÇEKTEN hesaplanır, varsayılmaz.

## Dayanıklılık

`{attacks_dir}/attack_results.json`'a her (saldırı,koşul) sonrası
ATOMİK yazılır (`scripts.bench_circuit.save_bench_results` — tmp+
`os.replace`); `--force` verilmedikçe zaten sonuçlanmış (saldırı,koşul)
çiftleri ATLANIR (Colab kopmalarına karşı devam edilebilir). Büyük ara
tensörler (`site_states`, üretilen görüntüler) her adımdan sonra
`del`/`gc.collect()`/`torch.cuda.empty_cache()` ile temizlenir.

BİLİNÇLİ TEST SINIRI: gerçek `raw_root` pkl'leri, StyleGAN-XL reposu,
GPU, gerçek APTOS/DR veri kümesi gerektirir — SADECE Colab'da çalışır.
Saf yardımcılar (`apply_attack`, `resolve_training_options_path`)
`tests/test_run_attacks.py`'de yerelde test edilir.

## CUDA custom op derlemesi başarısız olursa (`--ops-impl`)

StyleGAN-XL'in `bias_act`/`upfirdn2d`/`filtered_lrelu` CUDA-derlemeli
custom op'ları bazı ortamlarda (ör. Colab'ın Python 3.13/güncel
PyTorch kombinasyonu) DERLENEMEZ (`ModuleNotFoundError: No module
named 'bias_act_plugin'`). `--ops-impl auto` (varsayılan) bunu KÜÇÜK
bir deneme üretimiyle tespit edip GEREKİRSE `eval.metrics.force_stylegan_ops_impl("ref")`
ile saf PyTorch referans uygulamasına (matematiksel olarak AYNI,
sadece YAVAŞ) düşer — bu düşüş konsola UYARIYLA loglanır, sessizce
geçilmez. Sonuç JSON'unda hangi modun kullanıldığı `ops_impl` alanında
kayıtlıdır. `ref` modunda ilk koşuldan sonra `--metrics-mode full`'un
tahmini maliyeti (saat cinsinden) otomatik loglanır.

## CUDA bellek yetersizliği (`--gen-batch-size`, `--device cpu`)

`impl='ref'` moduna düşüldüğünde `upfirdn2d`'nin saf PyTorch yolu
StyleGAN3-r'ın geniş filtreleriyle TEK bir görüntü için bile onlarca
GB'lık ara tensör üretebiliyor (gerçek Colab koşumunda gözlendi:
`F.pad`'de 13.37 GiB tek seferlik istek). `--gen-batch-size` görüntü
üretim batch boyutunu elle ayarlar (varsayılan: `ops_impl`'e göre
otomatik — `ref` için 1, `cuda` için 32); `eval.metrics.generate_class_images`
OOM'da batch boyutunu KENDİLİĞİNDEN yarıya indirip yeniden dener,
batch 1'de bile OOM olursa net bir hata + `--device cpu` önerisiyle
durur. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (hata
mesajının önerdiği, bellek parçalanmasını azaltan ayar) `torch`
import edilmeden ÖNCE burada ayarlanıyor.

Kullanım (Colab'da):
    python -m scripts.run_attacks --env colab --round 10 --metrics-mode quick    # önce ucuz sağlama
    python -m scripts.run_attacks --env colab --round 10 --metrics-mode full     # resmi fid50k_full/kid50k_full ('cuda' çalışıyorsa)
    python -m scripts.run_attacks --env colab --round 10 --metrics-mode custom --fid-num-gen 5000  # 'ref'te 'full' pratik değilse
    python -m scripts.run_attacks --env colab --round 10 --ops-impl ref         # CUDA derlemesini hiç deneme, doğrudan ref
    python -m scripts.run_attacks --env colab --round 10 --device cpu          # GPU'da batch=1 bile OOM verirse
"""

from __future__ import annotations

import os

# torch import edilmeden ÖNCE ayarlanmalı (CUDA ayırıcısı ilk tahsiste
# bu değeri okuyor) — hata mesajının önerdiği, bellek parçalanmasını
# azaltan ayar. Kullanıcı zaten farklı bir değer ayarladıysa (ör.
# notebook'ta elle) EZİLMEZ (`setdefault`).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402 - PYTORCH_CUDA_ALLOC_CONF ayarından SONRA olmalı
import gc  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from attacks.conditional_poison import DEFAULT_EMBED_KEY, swap_embed_rows  # noqa: E402
from attacks.detection import attack_touches_zk_proven_scope, verify_commitment_consistency
from attacks.random_weights import randomize_state_dict
from attacks.scaled_poison import scale_delta
from configs.loader import load_paths
from eval.metrics import DEFAULT_CUSTOM_FID_NUM_GEN
from orchestrator.aggregate import aggregate_round, compute_delta_norm
from orchestrator.schedule import load_schedule_config
from scripts.bench_circuit import (
    load_bench_results as load_attack_results,
    save_bench_results as save_attack_results,
    write_json_file,
)
from scripts.inventory import find_run_subdir
from storage.pathguard import assert_writable

DEFAULT_ROUND = 10
DEFAULT_POISONED_SITE_INDEX = 0
DR_NUM_CLASSES = 5
SWAP_ROW_A, SWAP_ROW_B = 0, 4  # DR-0 / DR-4 (kullanıcının belirttiği saldırı hedefi)

ATTACK_NAMES = ("random_weights", "scaled_poison_10x", "scaled_poison_50x", "scaled_poison_100x", "conditional_poison")
CONDITIONS = ("unprotected", "protected", "poisoned_alone")

FULL_CLASS_IMAGES_PER_CLASS = 200
QUICK_CLASS_IMAGES_PER_CLASS = 20

# ops_impl'e göre varsayılan görüntü üretim batch boyutu — 'ref' modunda
# StyleGAN3-r'ın geniş filtreleriyle upfirdn2d'nin saf PyTorch yolu TEK
# bir görüntü için bile devasa ara tensör üretebiliyor (gerçek Colab
# koşumunda gözlendi: F.pad'de 13.37 GiB), bu yüzden 'ref' için EN
# KÜÇÜK (1) batch varsayılan; 'cuda' fused kernel kullandığından daha
# büyük batch güvenli.
DEFAULT_GEN_BATCH_SIZE_BY_OPS_IMPL = {"cuda": 32, "ref": 1}


# ---------------------------------------------------------------------------
# Saf yardımcılar (StyleGAN-XL/GPU GEREKTİRMEZ)
# ---------------------------------------------------------------------------


def result_key(attack_name: str, condition: str) -> str:
    return f"{attack_name}__{condition}"


def pick_default_gen_batch_size(ops_impl: str) -> int:
    """`--gen-batch-size` elle verilmediğinde `ops_impl`'e göre GÜVENLİ
    bir varsayılan seçer — `ref` için 1 (bkz. `DEFAULT_GEN_BATCH_SIZE_BY_OPS_IMPL`
    docstring'i), `cuda` için 32."""
    if ops_impl not in DEFAULT_GEN_BATCH_SIZE_BY_OPS_IMPL:
        raise ValueError(f"Bilinmeyen ops_impl: {ops_impl!r} (geçerli: {sorted(DEFAULT_GEN_BATCH_SIZE_BY_OPS_IMPL)})")
    return DEFAULT_GEN_BATCH_SIZE_BY_OPS_IMPL[ops_impl]


def resolve_training_options_path(raw_root: str, round_id: int, site_id: int) -> str:
    """`training_options.json`'un GERÇEK yolunu, `scripts.inventory.find_run_subdir`
    (zaten test edilmiş) ile çözer — Faz A'nın inventory taramasıyla
    AYNI dizin yapısı varsayımı, hiçbir yeni yol icat edilmez."""
    site_dir = os.path.join(raw_root, f"round_{round_id}", f"site_{site_id}")
    if not os.path.isdir(site_dir):
        raise FileNotFoundError(f"Site dizini bulunamadı: '{site_dir}'.")
    run_dir, warning = find_run_subdir(site_dir)
    if run_dir is None:
        raise RuntimeError(f"round {round_id} site {site_id}: {warning}")
    path = os.path.join(run_dir, "training_options.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"'{path}' bulunamadı — round {round_id} site {site_id}.")
    return path


def apply_attack(attack_name: str, *, poisoned_site_state: dict, prev_global_state: dict) -> tuple[dict, list]:
    """Saldırıyı `poisoned_site_state`'e uygular, `(poisoned_state,
    modified_keys)` döner. `modified_keys`: saldırının KAVRAMSAL
    olarak hedeflediği anahtar kümesi (ZK kapsam analizi için —
    `attacks/detection.py: attack_touches_zk_proven_scope`)."""
    if attack_name == "random_weights":
        poisoned = randomize_state_dict(poisoned_site_state, generator=torch.Generator().manual_seed(0))
        return poisoned, list(poisoned.keys())

    if attack_name.startswith("scaled_poison_"):
        scale_str = attack_name.removeprefix("scaled_poison_").removesuffix("x")
        scale = float(scale_str)
        poisoned = scale_delta(poisoned_site_state, prev_global_state, scale=scale)
        return poisoned, list(poisoned.keys())

    if attack_name == "conditional_poison":
        poisoned = swap_embed_rows(poisoned_site_state, SWAP_ROW_A, SWAP_ROW_B, embed_key=DEFAULT_EMBED_KEY)
        return poisoned, [DEFAULT_EMBED_KEY]

    raise ValueError(f"Bilinmeyen saldırı: {attack_name!r} (geçerli: {ATTACK_NAMES})")


def build_unprotected_global(site_states: dict) -> dict:
    """Hiçbir kapı YOK — 4 site (3 dürüst + 1 zehirli) doğrudan
    ortalanır. `fl.fedavg_utils.RunningAverage` YENİDEN YAZILMAZ."""
    from fl.fedavg_utils import RunningAverage

    running = RunningAverage()
    for state in site_states.values():
        running.update(state)
    return running.result()


def estimate_full_mode_cost(seconds_per_image: float, *, num_conditions: int = 10, images_per_metric_run: int = 50000, metrics_per_condition: int = 2) -> dict:
    """`impl='ref'`e düşüldüğünde `--metrics-mode full`'un (`fid50k_full`
    `+ kid50k_full`, İKİSİ de 50.000 görüntü ÜRETİR — WebFetch ile
    doğrulanan resmi ayar) gerçekçi bir maliyet TAHMİNİ — `seconds_per_image`
    GERÇEK ölçümden (`class_confusion_matrix`'in `generation_seconds`/
    `num_generated_images`'ından) geliyor, UYDURULMUYOR. `num_conditions=10`:
    5 saldırı × 3 koşul (bu script'in TAM koşumu, `poisoned_alone` dahil)."""
    if seconds_per_image <= 0:
        raise ValueError(f"seconds_per_image pozitif olmalı, alınan: {seconds_per_image}")
    seconds_per_metric_run = seconds_per_image * images_per_metric_run
    seconds_per_condition = seconds_per_metric_run * metrics_per_condition
    return {
        "seconds_per_image": seconds_per_image,
        "hours_per_condition": seconds_per_condition / 3600.0,
        "hours_total": (seconds_per_condition * num_conditions) / 3600.0,
        "num_conditions": num_conditions,
    }


def build_protected_global(prev_global: dict, site_states: dict, *, tau_norm_threshold: float) -> dict:
    """`orchestrator.aggregate.aggregate_round` DOĞRUDAN çağrılır —
    ZK-onay (`chain_approved`) + norm kapısının İKİSİ de zaten orada.
    `chain_approved=True` HER site için: bu simülasyondaki "dürüst ama
    bozulmuş" istemci modelinde ZK her zaman geçerdi (bkz. modül
    docstring'i, madde 5)."""
    chain_approved = {site: True for site in site_states}
    return aggregate_round(prev_global, site_states, tau_norm_threshold=tau_norm_threshold, chain_approved=chain_approved)


def build_poisoned_alone_global(poisoned_state: dict, poisoned_site) -> dict:
    """H1 TANI KOŞULU — FedAvg'ı TAMAMEN ATLAYIP zehirli site'ın
    KENDİ (tek başına) ağırlığını "global" olarak kullanır. Gerçek
    Colab bulgusu: `conditional_poison`'ın DR-0/DR-4 takası FedAvg'lı
    (4 site, zehirli site 1/4 ağırlıkla) sonuç matrisinde GÖRÜNMEDİ
    (köşegen korunmuş). Bu, saldırının 3 dürüst site tarafından
    SEYRELTİLDİĞİ hipotezini (H1) doğrudan test eder — bu koşulda
    seyreltme YOK (tek site, ağırlık 1.0), takas hâlâ görünmüyorsa H1
    ELENİR ve sorun gömme takasının kendisinde (H2) ya da ölçüm
    duyarlılığında (H3) aranmalı."""
    return {"global_state": poisoned_state, "included_sites": [poisoned_site], "gate_results": None}


# ---------------------------------------------------------------------------
# StyleGAN-XL/GPU/gerçek veri kümesine bağımlı adımlar
# (BİLİNÇLİ TEST SINIRI — sadece Colab'da çalışır)
# ---------------------------------------------------------------------------


def resolve_ops_impl(ops_impl_arg: str, *, g_ema, z_dim: int, c_dim: int, device) -> str:
    """`--ops-impl`e göre StyleGAN-XL'in CUDA-derlemeli custom
    op'larının (`bias_act`/`upfirdn2d`/`filtered_lrelu`) hangi
    uygulamayla (`cuda`/`ref`) çalışacağına karar verir.

    `"cuda"`/`"ref"`: `eval.metrics.force_stylegan_ops_impl` ile
    DOĞRUDAN zorlanır. `"auto"` (varsayılan): ÖNCE CUDA plugin
    derlemesini KÜÇÜK bir deneme üretimiyle (batch_size=1) test eder;
    BAŞARISIZ olursa (`bias_act_plugin` gibi bir derleme hatası —
    Colab'ın Python 3.13/güncel PyTorch ortamında BİLİNEN bir sorun)
    `ref`'e DÜŞER. Bu düşüş SESSİZCE geçilmez, net bir UYARI basılır
    (kullanıcının açık isteği) — `--ops-impl auto` ile hangi ortamda
    çalıştığı ÖNCEDEN bilinmeden de doğru moda GERÇEKTEN karar
    verilir, varsayılmaz. `device.type == "cpu"` ise smoke-test HİÇ
    yapılmaz — StyleGAN-XL'in kendi `impl` dallanması
    (`if impl=='cuda' and x.device.type=='cuda' and _init(): ...`,
    WebFetch ile doğrulandı) zaten CPU tensörlerinde `impl` DEĞERİNDEN
    BAĞIMSIZ olarak ref yoluna düşüyor — `ops_impl='ref'` DOĞRUDAN
    döndürülür (test etmeye gerek yok, sonuç zaten kesin).

    **Gerçek Colab bulgusu (düzeltildi):** smoke-test önceden `g_ema`'yı
    `device`'a TAŞIMADAN çalıştırılıyordu — `z`/`c` GPU'ya taşınırken
    `g_ema` CPU'da kalınca `RuntimeError: Expected all tensors to be
    on the same device (... wrapper_CUDA__index_select)` ile
    çöküyordu; bu bir CUDA DERLEME hatası DEĞİLDİ (bir kod hatasıydı),
    ama eski kod bunu "derleme başarısız" sanıp SESSİZCE `ref`'e
    düşüyordu — CUDA aslında ÇALIŞIYOR olabilirdi. Artık `g_ema.to(device)`
    smoke-test'ten ÖNCE çağrılıp `assert_module_on_device` ile
    doğrulanıyor; `is_device_mismatch_error` gerçek bir cihaz hatasını
    derleme hatasından AYIRT edip AYRI, net bir hata olarak yükseltiyor
    (sessizce ref'e düşürmüyor — bu sınıf hata bir daha GİZLENMESİN)."""
    from eval.metrics import assert_module_on_device, force_stylegan_ops_impl, generate_class_images, is_device_mismatch_error

    if device.type == "cpu":
        print("[run_attacks] device=cpu: CUDA custom op'ları CPU tensörlerinde zaten devre dışı — ops_impl='ref' (smoke-test atlandı).")
        return "ref"

    if ops_impl_arg in ("cuda", "ref"):
        force_stylegan_ops_impl(ops_impl_arg)
        return ops_impl_arg

    print("[run_attacks] --ops-impl auto: CUDA custom op derlemesi küçük bir deneme üretimiyle test ediliyor...")
    g_ema.to(device)
    assert_module_on_device(g_ema, device)
    try:
        generate_class_images(g_ema, class_index=0, c_dim=c_dim, z_dim=z_dim, num_images=1, device=device, batch_size=1)
        print("[run_attacks] --ops-impl auto: CUDA custom op'ları ÇALIŞIYOR, 'cuda' kullanılacak.")
        return "cuda"
    except Exception as e:  # noqa: BLE001 - CUDA derleme hatasi COK CESITLI olabilir (ModuleNotFoundError, RuntimeError, ninja hatasi...), hepsi ayni sekilde ref'e dusurulmeli - CIHAZ hatasi AYRI ele alinir (asagida)
        if is_device_mismatch_error(e):
            raise RuntimeError(
                f"[run_attacks] --ops-impl auto smoke-test'i bir CİHAZ UYUŞMAZLIĞI hatasıyla çöktü: {e} — "
                f"bu bir CUDA DERLEME hatası DEĞİL, kodda bir cihaz yerleştirme hatası (regresyon). "
                f"'ref'e sessizce düşülmüyor — kaynağı bulunup düzeltilmeli."
            ) from e
        print(
            f"[run_attacks] UYARI: --ops-impl auto: CUDA custom op derlemesi BAŞARISIZ "
            f"({type(e).__name__}: {e}) — 'ref' (saf PyTorch referans) uygulamasına DÜŞÜLÜYOR. "
            f"Bu, StyleGAN-XL'in CUDA custom ops'unun bu ortamda (ör. Python 3.13/güncel PyTorch) "
            f"DERLENEMEDİĞİ bilinen bir durum — matematiksel sonuç AYNI, sadece daha YAVAŞ."
        )
        force_stylegan_ops_impl("ref")
        return "ref"


def evaluate_condition(
    *,
    global_state: dict,
    g_ema_shell,
    dims: dict,
    metric_options: dict,
    dataset_root_override: str | None,
    metrics_mode: str,
    device,
    gen_batch_size: int,
    fid_num_gen: int,
) -> dict:
    """Bir (saldırı,koşul) çiftinin GERÇEK global ağırlığını `g_ema_shell`'e
    yükleyip FID/KID (`--metrics-mode full`/`custom`) + sınıf-tutarlılığı
    matrisini hesaplar. `custom` modu — `ref`'te `full`'un 50k
    görüntüsü pratik OLMADIĞINDA — `eval.metrics.run_custom_fid_kid`
    ile YAPILANDIRILABİLİR (`fid_num_gen`) bir örnekle GERÇEK bir
    FID/KID hesaplar, ama sonucu `13.13` referansıyla KIYASLANAMAZ
    olarak (`fid_comparable_to_reference=False`) İŞARETLER."""
    from eval.metrics import class_confusion_matrix, run_custom_fid_kid, run_official_metric

    t0 = time.perf_counter()
    g_ema_shell.load_state_dict(global_state)
    g_ema_shell.to(device).eval()

    result: dict = {}
    if metrics_mode == "full":
        for metric_name in ("fid50k_full", "kid50k_full"):
            t_metric = time.perf_counter()
            metric_result = run_official_metric(
                metric_name, g_ema_shell, dataset_kwargs=metric_options["dataset_kwargs"],
                num_gpus=metric_options["num_gpus"], device=device,
            )
            result.update(metric_result)
            result[f"{metric_name}_seconds"] = time.perf_counter() - t_metric
            print(f"[run_attacks]   {metric_name}: {metric_result} ({result[f'{metric_name}_seconds']:.1f}s)")
    elif metrics_mode == "custom":
        t_metric = time.perf_counter()
        custom_result = run_custom_fid_kid(
            g_ema_shell, dataset_kwargs=metric_options["dataset_kwargs"], num_gpus=metric_options["num_gpus"],
            device=device, num_gen=fid_num_gen,
        )
        result.update(custom_result)
        print(
            f"[run_attacks]   ÖZEL FID/KID (num_gen={fid_num_gen}, 13.13 İLE KARŞILAŞTIRILAMAZ): "
            f"fid_custom={custom_result['fid_custom']:.3f} kid_custom={custom_result['kid_custom']:.5f} "
            f"({time.perf_counter() - t_metric:.1f}s)"
        )
    else:
        print("[run_attacks]   --metrics-mode quick: resmi fid50k_full/kid50k_full ATLANDI (sadece sınıf-tutarlılığı koşuluyor).")

    num_images_per_class = QUICK_CLASS_IMAGES_PER_CLASS if metrics_mode == "quick" else FULL_CLASS_IMAGES_PER_CLASS
    t_confusion = time.perf_counter()
    confusion = class_confusion_matrix(
        g_ema_shell, dataset_kwargs=metric_options["dataset_kwargs"], num_classes=DR_NUM_CLASSES,
        z_dim=dims["z_dim"], device=device, num_images_per_class=num_images_per_class, batch_size=gen_batch_size,
    )
    result["class_confusion"] = confusion
    result["class_confusion_seconds"] = time.perf_counter() - t_confusion
    result["mean_diagonal_kid"] = sum(confusion["kid_matrix"][i][i] for i in range(DR_NUM_CLASSES)) / DR_NUM_CLASSES
    result["total_seconds"] = time.perf_counter() - t0

    del global_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz F: saldırı × koruma matrisi (FID/KID/sınıf-tutarlılığı).")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--schedule-config", default="configs/schedule.yaml")
    parser.add_argument("--round", type=int, default=DEFAULT_ROUND, help="ağırlıkların alınacağı gerçek round (--round - 1'in fedavg dosyası önceki global olarak kullanılır)")
    parser.add_argument("--poisoned-site", type=int, default=DEFAULT_POISONED_SITE_INDEX)
    parser.add_argument("--dataset-root", default=None, help="training_options.json'daki dataset yolu bu makinede geçersizse, yeniden konumlandırma kökü")
    parser.add_argument("--attacks-dir", default=None, help="varsayılan: {zk_root}/attacks")
    parser.add_argument(
        "--metrics-mode", default="full", choices=["full", "quick", "custom"],
        help="'full': resmi fid50k_full/kid50k_full (saatler sürebilir, 5 saldırı x 3 koşul x 50k görüntü — "
             "'ref' modunda GENELLİKLE PRATİK DEĞİL, bkz. koşum başındaki maliyet ekstrapolasyonu). "
             "'quick': FID/KID ATLANIR, sadece küçük örnekli sınıf-tutarlılığı koşulur (dakikalar, önce sağlama için). "
             "'custom': GERÇEK ama küçük örnekli (--fid-num-gen) bir FID/KID — 13.13 İLE KARŞILAŞTIRILAMAZ, "
             "SADECE bu koşumun kendi saldırı×koruma karşılaştırması için.",
    )
    parser.add_argument(
        "--fid-num-gen", type=int, default=None,
        help=f"'--metrics-mode custom' için üretilecek görüntü sayısı (varsayılan: {DEFAULT_CUSTOM_FID_NUM_GEN}). "
             f"13.13 referansı 50000 ile ölçüldü — bu değer KÜÇÜLTÜLDÜKÇE karşılaştırılamazlık ARTAR.",
    )
    parser.add_argument("--only-attack", default=None, help="virgülle ayrılmış saldırı adı listesi — sadece bunları koştur (bkz. ATTACK_NAMES)")
    parser.add_argument("--force", action="store_true", help="zaten sonuçlanmış (saldırı,koşul) çiftlerini yeniden çalıştır")
    parser.add_argument(
        "--ops-impl", default="auto", choices=["auto", "cuda", "ref"],
        help="StyleGAN-XL custom op'ları (bias_act/upfirdn2d/filtered_lrelu). 'auto' (varsayılan): CUDA'yı dener, "
             "derleme başarısızsa UYARIYLA 'ref'e düşer. 'cuda'/'ref': doğrudan zorla.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="'auto' (varsayılan): CUDA varsa kullan. 'cpu': GPU'da batch=1 bile CUDA OOM verirse fallback "
             "(çok daha yavaş — süre ölçülüp loglanır).",
    )
    parser.add_argument(
        "--gen-batch-size", type=int, default=None,
        help="Görüntü üretim batch boyutu. Varsayılan: ops_impl'e göre otomatik (ref=1, cuda=32) — "
             "bkz. DEFAULT_GEN_BATCH_SIZE_BY_OPS_IMPL.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from eval.metrics import load_metric_options_from_training_options
    from fl.module_tree import extract_known_attrs
    from fl.round_replay import load_site_g_ema_module, load_site_update
    from scripts.audit_fedavg import load_fedavg_file

    paths = load_paths(env=args.env, config_path=args.paths_config)
    schedule_config = load_schedule_config(args.schedule_config)
    tau = schedule_config["tau_norm_threshold"]

    attacks_dir = args.attacks_dir or os.path.join(paths["zk_root"], "attacks")
    assert_writable(attacks_dir)
    os.makedirs(attacks_dir, exist_ok=True)
    results_path = os.path.join(attacks_dir, "attack_results.json")
    all_results = load_attack_results(results_path)

    attack_names = [a.strip() for a in args.only_attack.split(",")] if args.only_attack else list(ATTACK_NAMES)
    unknown = [a for a in attack_names if a not in ATTACK_NAMES]
    if unknown:
        raise ValueError(f"Bilinmeyen saldırı adı/ları: {unknown}. Geçerli: {ATTACK_NAMES}")

    fid_num_gen = args.fid_num_gen or DEFAULT_CUSTOM_FID_NUM_GEN
    print(f"[run_attacks] round={args.round} poisoned_site={args.poisoned_site} attacks={attack_names} metrics_mode={args.metrics_mode} fid_num_gen={fid_num_gen}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[run_attacks] device={device}")

    print("[run_attacks] Gerçek site state_dict'leri yükleniyor (Faz A checkpoint'leri, salt okunur)...")
    site_states_original = {i: load_site_update(paths["raw_root"], paths["stylegan_xl_repo"], args.round, i) for i in range(4)}

    prev_global_path = os.path.join(paths["raw_root"], f"fedavg_{args.round - 1}.pt")
    print(f"[run_attacks] Önceki global yükleniyor: {prev_global_path}")
    prev_global_state = load_fedavg_file(prev_global_path)

    print("[run_attacks] G_ema 'shell' modülü yükleniyor (state_dict'i her koşulda değiştirilecek)...")
    g_ema_shell = load_site_g_ema_module(paths["raw_root"], paths["stylegan_xl_repo"], args.round, 0)
    dims_result = extract_known_attrs(g_ema_shell, names=("z_dim", "c_dim", "w_dim", "num_ws"))
    if dims_result["bulunamadi"]:
        raise RuntimeError(f"G_ema üzerinde beklenen boyut alanları bulunamadı: {dims_result['bulunamadi']}")
    dims = dims_result["found"]
    if dims["c_dim"] != DR_NUM_CLASSES:
        raise RuntimeError(f"Beklenen c_dim={DR_NUM_CLASSES}, gerçek G_ema.c_dim={dims['c_dim']} — DR sınıf sayısı varsayımı YANLIŞ, sabitleri güncelle.")
    print(f"[run_attacks] Boyutlar: {dims}")

    ops_impl_used = resolve_ops_impl(args.ops_impl, g_ema=g_ema_shell, z_dim=dims["z_dim"], c_dim=dims["c_dim"], device=device)
    gen_batch_size = args.gen_batch_size or pick_default_gen_batch_size(ops_impl_used)
    print(f"[run_attacks] ops_impl={ops_impl_used} gen_batch_size={gen_batch_size}")
    extrapolation_logged = False

    training_options_path = resolve_training_options_path(paths["raw_root"], args.round, 0)
    print(f"[run_attacks] Ölçüm ayarları okunuyor: {training_options_path}")
    metric_options = load_metric_options_from_training_options(training_options_path, dataset_root_override=args.dataset_root)
    print(f"[run_attacks] metrics={metric_options['metrics']} num_gpus={metric_options['num_gpus']} dataset_path={metric_options['dataset_kwargs'].get('path')}")

    for attack_name in attack_names:
        poisoned_site_original = site_states_original[args.poisoned_site]
        poisoned_state, modified_keys = apply_attack(attack_name, poisoned_site_state=poisoned_site_original, prev_global_state=prev_global_state)

        delta_norm = compute_delta_norm(prev_global_state, poisoned_state)
        norm_caught = delta_norm > tau
        zk_caught = not verify_commitment_consistency(poisoned_state, poisoned_state)
        zk_in_scope = attack_touches_zk_proven_scope(modified_keys)
        print(
            f"\n=== Saldırı: {attack_name} — ||ΔG||={delta_norm:.2f} (tau={tau}) "
            f"norm_caught={norm_caught} zk_in_scope={zk_in_scope} zk_caught={zk_caught} ==="
        )

        site_states = dict(site_states_original)
        site_states[args.poisoned_site] = poisoned_state

        for condition in CONDITIONS:
            key = result_key(attack_name, condition)
            if key in all_results and not args.force:
                print(f"[run_attacks] '{key}' zaten sonuçlanmış, atlanıyor (--force ile yeniden çalıştır).")
                continue

            print(f"[run_attacks] --- koşul: {condition} ---")
            if condition == "unprotected":
                aggregation = {"global_state": build_unprotected_global(site_states), "included_sites": list(site_states), "gate_results": None}
            elif condition == "protected":
                aggregation = build_protected_global(prev_global_state, site_states, tau_norm_threshold=tau)
            else:  # "poisoned_alone" — H1 tanı koşulu, bkz. build_poisoned_alone_global docstring'i
                aggregation = build_poisoned_alone_global(poisoned_state, args.poisoned_site)

            poisoned_included = args.poisoned_site in aggregation["included_sites"]
            print(f"[run_attacks] '{key}': zehirli site dahil mi? {poisoned_included}")

            metrics_result = evaluate_condition(
                global_state=aggregation["global_state"], g_ema_shell=g_ema_shell, dims=dims,
                metric_options=metric_options, dataset_root_override=args.dataset_root,
                metrics_mode=args.metrics_mode, device=device, gen_batch_size=gen_batch_size,
                fid_num_gen=fid_num_gen,
            )

            all_results[key] = {
                "attack": attack_name,
                "condition": condition,
                "ops_impl": ops_impl_used,
                "device": str(device),
                "gen_batch_size": gen_batch_size,
                "poisoned_site_included": poisoned_included,
                "delta_norm": delta_norm,
                "norm_caught": norm_caught,
                "zk_in_scope": zk_in_scope,
                "zk_caught": zk_caught,
                "modified_keys_count": len(modified_keys),
                **metrics_result,
            }
            save_attack_results(all_results, results_path)
            print(f"[run_attacks] '{key}' tamamlandı, {results_path} güncellendi.")

            if not extrapolation_logged:
                extrapolation_logged = True
                confusion = metrics_result["class_confusion"]
                seconds_per_image = confusion["generation_seconds"] / confusion["num_generated_images"]
                print(f"[run_attacks] Üretim hızı (ops_impl={ops_impl_used}, device={device}): {seconds_per_image:.3f}s/görüntü ({confusion['num_generated_images']} görüntü, {confusion['generation_seconds']:.1f}s).")
                if ops_impl_used == "ref" or device.type == "cpu":
                    cost = estimate_full_mode_cost(seconds_per_image, num_conditions=len(attack_names) * len(CONDITIONS))
                    print(
                        f"[run_attacks] UYARI: '{ops_impl_used}'/{device} modunda --metrics-mode full tahmini maliyet: "
                        f"koşul başına ~{cost['hours_per_condition']:.1f} saat, "
                        f"TOPLAM ({cost['num_conditions']} koşul) ~{cost['hours_total']:.1f} saat "
                        f"(SADECE görüntü üretimi — Inception özellik çıkarımı/KID hesaplaması HARİÇ, ek zaman gerektirir). "
                        f"Bu pratik olmayabilir — alternatifler: (a) --only-attack ile saldırı sayısını azalt, "
                        f"(b) '--metrics-mode custom --fid-num-gen {DEFAULT_CUSTOM_FID_NUM_GEN}' ile GERÇEK ama küçük "
                        f"örnekli bir FID/KID (13.13 İLE KARŞILAŞTIRILAMAZ, ama bu koşumun kendi saldırı×koruma "
                        f"karşılaştırması için yeterli), (c) CUDA custom op derlemesini bu Colab ortamında (farklı "
                        f"PyTorch/CUDA sürümü, ör. `pip install torch==<uyumlu sürüm>`) çalışır hale getirmeyi dene."
                    )

        del poisoned_state, site_states
        gc.collect()

    print(f"\n[run_attacks] Bitti. Tam sonuç: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
