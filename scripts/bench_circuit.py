#!/usr/bin/env python
"""Faz C3: gerçek mapping devresi için ezkl benchmark ızgarası.

Izgara boyutları: k in {1,4,8} x embed_mode in {matmul,gather} x
scale in {6,7,8,9,10} (varsayılanlar; hepsi CLI'dan konfigüre edilebilir).
Faz C2'nin ürettiği `{onnx_dir}/mapping_k{k}_{embed_mode}.onnx`
dosyalarını girdi alır (bkz. circuits/export_mapping.py).

## Ölçek ızgarası neden {6,7,8,9,10} (eskiden {8,11,13})

`scripts/diagnose_calibration.py`'nin ilk Colab koşumu üç net bulgu
verdi (bkz. `docs/phase_c_calibration.md`):
1. `calibrate_settings(..., scales=[scale])` kwarg'ı GERÇEKTEN çalışıyor
   (scale=8, target=resources) — `max_abs_error=0.133`, `logrows=19`.
2. **scale=11 ve scale=13 DÜŞTÜ** ("[halo2] General synthesis error",
   "[tensor] significant bit truncation ... try lowering the scale") —
   beklentinin TERSİ: DÜŞÜK ölçek çalışıyor, YÜKSEK ölçek düşüyor.
   Muhtemel sebep: `mapping.fc1`'in 512 terimlik iç çarpımları yüksek
   ölçekte decomposition tabanını aşıyor.
3. scale=16'da ezkl'in Rust tarafı gerçek bir `pyo3_runtime.PanicException`
   fırlattı (BaseException'dan türer, `except Exception` YAKALAMAZ).

Bu yüzden ızgara artık ÇALIŞAN sınırın (scale=8) etrafında ince bir
tarama ({6,7,8,9,10}) yapıyor — amaç: en yüksek ÇALIŞAN scale'i bulmak
(yüksek scale = daha iyi fidelity, ama bu devrede bir yerde kesiliyor).

## Tasarım: her kombinasyon AYRI bir alt süreçte çalışır

Üç bağımsız gerekçe var:
1. **Tepe RAM ölçümü kombinasyon başına.** `resource.getrusage(RUSAGE_SELF).ru_maxrss`
   bir sürecin YAŞAM BOYU yüksek-su-işareti değeridir, geriye sarılamaz.
   Aynı süreçte art arda kombinasyon koşulursa N. kombinasyonun kendi
   tepe RAM'ini önceki kombinasyonlardan ayıramayız. Her kombinasyon
   kendi alt sürecinde çalışırsa, o sürecin `ru_maxrss`'i TAM O
   kombinasyonun tepe RAM'idir.
2. **ezkl'nin Rust tarafının bastığı loglar (decomposition uyarıları,
   "Numerical Fidelity Report" / `max_abs_error`) Python `sys.stdout`
   yönlendirmesiyle (`contextlib.redirect_stdout`) YAKALANAMAYABİLİR**
   (pyo3 bağlaması gerçek OS dosya tanıtıcısına yazıyor olabilir). Alt
   süreç yaklaşımıyla ebeveyn, `subprocess.PIPE` ile alt sürecin GERÇEK
   stdout/stderr fd'sini yakalar — Python içi yönlendirmeden bağımsız,
   güvenilir.
3. **Çökme izolasyonu.** ezkl/anvil bir kombinasyonda çökerse (segfault,
   OOM) sadece o alt süreç ölür, ebeveyn ayakta kalıp diğer
   kombinasyonlara devam eder ("bir kombinasyon patlarsa diğerlerine
   devam et" gereksinimi).

## Sabit scale: `calibrate_settings(..., scales=[scale])` (DOĞRULANDI)

Önceki tur (`force_settings_scale`: calibrate SONRASI settings.json'u
elle yamama) `scripts/diagnose_calibration.py`'nin Colab koşumunda
**hiçbir ölçekte tutmadı** — `compile_circuit` hâlâ calibrate'in
seçtiği scale'i kullanıyordu. Doğru/çalışan yöntem, `ezkl==23.0.5`'in
(`v23.0.5` etiketi) `ezkl.pyi`'sinde belgelenen gerçek `scales:
Optional[Sequence[int]]` parametresini DOĞRUDAN `calibrate_settings`'e
vermek — bu Colab'da ampirik olarak ÇALIŞTI (scale=8, target=resources).
`PyRunArgs.input_scale`/`param_scale` bu yüzden ARTIK gen_settings'te
verilmiyor (calibrate zaten `scales=[scale]` ile scale'i kontrol
ediyor); calibrate SONRASI gerçekleşen `input_scale`/`param_scale`/
`logrows` `read_realized_scale` ile okunup raporlanıyor (istenen ile
gerçekleşen FARKLI olabilir, ikisi de kaydediliyor).

## Çoklu-girdi input.json formatı (DOĞRULANDI)

`{"input_data": [flat_z, flat_c]}` — `ezkl`'in `src/graph/input.rs`
kaynağından (`DataSource = Vec<Vec<FileSourceInner>>`) doğrulanmıştı;
diagnose_calibration'ın Colab koşumunda ONNX-şekli/input.json çapraz
kontrolü UYUŞTU ve kalibrasyon en azından scale=8'de gerçekten BAŞARILI
oldu — şema sorunu DEĞİLMİŞ, sorun scale seçimiymiş.

## Devre istatistikleri (best-effort, değişmedi)

`settings.json`'daki olası anahtarlar (`num_rows`, `run_args.logrows`,
`required_lookups` vb.) best-effort toplanır; şema sürümden sürüme
değişebileceğinden hiçbiri zorunlu değildir, ham `settings.json` dosyası
zaten `work_dir` altında kalıcı.

## pyo3 `PanicException` — `BaseException` olarak yakalanıyor

ezkl'nin Rust tarafı bazı (scale, devre) kombinasyonlarında GERÇEK bir
Rust panic'i `pyo3_runtime.PanicException` olarak fırlatabiliyor
(Colab'da scale=16'da gözlendi) — bu `BaseException`'dan türer,
`except Exception` YAKALAMAZ ve script'i düşürür. `run_worker`'ın dış
`try/except`'i bu yüzden `KeyboardInterrupt`/`SystemExit` HARİÇ tüm
`BaseException`'ı yakalar; panik özel olarak `status="panic"` ile
işaretlenir (sıradan `"failed"`'den ayrı — hangi kombinasyonların
ezkl'i gerçekten ÇÖKERTTİĞÜ, hangilerinin sadece normal bir hata
döndürdüğü karışmasın).

## Disk yönetimi: work_dir yerel diskte, sonuçlar Drive'da

İlk gerçek Colab koşumunda `pk.key` **2.5 GB** çıktı — 30 kombinasyonluk
tam ızgarada Drive kotasını doldurur. İki önlem alındı:
1. **Varsayılan çalışma kökü artık `/content` (Colab'ın yerel VM diski,
   Drive DEĞİL) altında** (`default_work_root()`), `--work-root` ile
   değiştirilebilir. Sadece küçük sonuç dosyaları (`{key}.json`,
   `{key}.log.txt`, `bench_results.json`) `--bench-dir`'e (varsayılan
   `{zk_root}/bench`, Drive) yazılıyor — büyük ara dosyalar (pk,
   derlenmiş devre, witness, settings, proof, Verifier.sol) hiç Drive'a
   gitmiyor.
2. Worker, ölçümleri (pk/vk boyutu dahil) `result.json`'a kaydettikten
   SONRA `cleanup_work_dir` ile büyük ara dosyaları (`pk.key`,
   `network.compiled`, `witness.json`) siler — boyutları zaten kayıtlı
   olduğundan dosyanın kalmasına gerek yok. `settings.json`/`proof.json`/
   `Verifier.sol`/`.abi`/`vk.key`/`input.json` KORUNUR (küçük, tekrar
   üretimi pahalı/gereksiz). `--keep-artifacts` verilirse HİÇBİR ŞEY
   silinmez (elle inceleme için). Her kombinasyondan sonra
   temizlik-öncesi/sonrası disk kullanımı loglanır ve `result.json`'a
   (`disk_usage_before_cleanup_bytes`/`_after_cleanup_bytes`/`_freed_bytes`)
   yazılır.

**NOT — SRS dosyası bu temizlik listesinde YOK:** `circuits/ezkl_utils.py:
run_get_srs` ve `run_ezkl_pipeline`/`run_prove_and_verify`'deki `setup`/
`gen_witness`/`prove` çağrıları `srs_path` argümanını hiç VERMİYOR
(`None` — ezkl.pyi'de `get_srs(settings_path, logrows, srs_path)` /
`setup(..., srs_path, ...)` şeklinde hepsi opsiyonel). Bu yüzden SRS
dosyası bizim `work_dir`'imize HİÇ yazılmıyor — ezkl kendi varsayılan
önbelleğini (VM'nin yerel diskinde, Drive'a gitmiyor) kullanıyor ve
AYNI `logrows` değerine sahip kombinasyonlar arasında PAYLAŞILIYOR
(yeniden üretilmiyor) — zaten bir sorun değil, silinecek bir şey yok.

BİLİNÇLİ TEST SINIRI: `run_ezkl_pipeline`/`run_prove_and_verify`/`run_worker`
ve zincir/solc adımları `ezkl`+`anvil`+`solc`+`web3` gerektirir, SADECE
Colab'da çalışır/test edilir (bu yüzden `ezkl`/zincire dokunan modüller
BURADA modül-seviyesinde DEĞİL, fonksiyon içinde import edilir — yerelde
`python -m scripts.bench_circuit --help` ve tüm saf yardımcı fonksiyonlar
yine de çalışır/test edilir). Izgara kurma, kombinasyon anahtarlama,
log ayrıştırma (decomposition uyarı sayısı, max_abs_error), devre
istatistik çıkarımı, input.json üretimi, ilerleme dosyası okuma/yazma
ve markdown tablo üretimi SAF fonksiyonlardır ve `tests/test_bench_circuit.py`'de
gerçekten test edilir.

Kullanım (Colab'da):
    python -m scripts.bench_circuit --env colab --only-first
    python -m scripts.bench_circuit --env colab
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

from circuits.ezkl_utils import run_async, run_get_srs
from circuits.export_mapping import load_reference, parse_k_values
from circuits.rebuild_mapping import VALID_EMBED_MODES
from configs.loader import load_paths
from storage.pathguard import assert_writable

try:
    import resource  # Sadece POSIX (Colab/Linux). Windows'ta yok.
except ImportError:  # pragma: no cover - yerelde (Windows) her zaman buraya düşer
    resource = None

DEFAULT_K_VALUES = (1, 4, 8)
DEFAULT_EMBED_MODES = ("matmul", "gather")  # matmul once: ArgMax/Gather'siz, DEFAULT_ONNX_MODE ile tutarli
# {8,11,13} degil {6,7,8,9,10}: diagnose_calibration.py'nin Colab bulgusu -
# scale=8 calisiyor, 11/13 dusuyor, 16 panikliyor. Calisan sinirin (8)
# etrafinda ince tarama yapip en yuksek CALISAN scale'i buluyoruz.
DEFAULT_SCALES = (6, 7, 8, 9, 10)
DEFAULT_TIMEOUT_SECONDS = 1800.0

INPUT_VISIBILITY = "private"
PARAM_VISIBILITY = "fixed"
OUTPUT_VISIBILITY = "public"

# Colab'da gercek mapping devresinde pk.key 2.5 GB cikti - boyutlari zaten
# result.json'a kaydedildiginden dosyanin kendisi gereksiz, 30 kombinasyonluk
# tam izgarada Drive kotasini doldurur. SRS burada YOK - work_dir'e hic
# yazilmiyor (bkz. modul docstring'i "Disk yonetimi" bolumu).
CLEANUP_DELETE_FILENAMES = ("pk.key", "network.compiled", "witness.json")

_MAX_ABS_ERROR_RE = re.compile(r"max_abs_error[\"']?\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")
_CIRCUIT_STAT_TOP_LEVEL_KEYS = ("num_rows", "num_constraints", "total_assignments")
_CIRCUIT_STAT_RUN_ARGS_KEYS = ("logrows", "input_scale", "param_scale", "num_inner_cols")


# ---------------------------------------------------------------------------
# Saf/testable yardımcılar (StyleGAN-XL/ezkl/anvil/solc GEREKTİRMEZ)
# ---------------------------------------------------------------------------


def build_grid(k_values: list[int], embed_modes: list[str], scales: list[int], only_first: bool = False) -> list[dict]:
    """Sıra ÖNEMLİ: k artan, embed_mode verilen sırayla, scale artan —
    ilk eleman her zaman (en küçük k, ilk mod, en düşük scale)."""
    for mode in embed_modes:
        if mode not in VALID_EMBED_MODES:
            raise ValueError(f"Geçersiz embed_mode: {mode!r} (geçerli: {VALID_EMBED_MODES})")
    grid = [{"k": k, "embed_mode": mode, "scale": scale} for k in k_values for mode in embed_modes for scale in scales]
    if only_first:
        return grid[:1]
    return grid


def combo_key(combo: dict) -> str:
    return f"k{combo['k']}_{combo['embed_mode']}_scale{combo['scale']}"


def count_decomposition_warnings(log_text: str) -> int:
    """Faz B'de gözlenen gerçek uyarı metni: "decomposition error: integer
    ... is too large" (bkz. docs/phase_b_report.md). Fatal değil, sınıra
    yaklaşmanın işareti."""
    return log_text.lower().count("decomposition error")


def extract_max_abs_error(log_text: str) -> float | None:
    """ezkl'nin "Numerical Fidelity Report"unda gözlenen `max_abs_error=...`
    alanını ayrıştırır (bkz. docs/phase_b_report.md: `max_abs_error=0.00033`).
    Birden fazla eşleşme varsa SONUNCUYU döner (en güncel rapor).
    Hiç bulunamazsa None (uydurmaz)."""
    matches = _MAX_ABS_ERROR_RE.findall(log_text)
    if not matches:
        return None
    return float(matches[-1])


def extract_circuit_stats(settings_json: dict) -> dict:
    """`settings.json`'dan olası devre istatistiklerini best-effort toplar.
    ezkl sürümleri arasında şema değişebileceğinden hiçbir anahtar zorunlu
    değildir — bulunanlar raporlanır, bulunamayanlar sessizce atlanır (ham
    settings.json zaten work_dir'da kalıcı, kayıp bilgi yok)."""
    stats: dict = {}
    if not isinstance(settings_json, dict):
        return stats
    run_args = settings_json.get("run_args", {})
    if isinstance(run_args, dict):
        for key in _CIRCUIT_STAT_RUN_ARGS_KEYS:
            if key in run_args:
                stats[key] = run_args[key]
    for key in _CIRCUIT_STAT_TOP_LEVEL_KEYS:
        if key in settings_json:
            stats[key] = settings_json[key]
    required_lookups = settings_json.get("required_lookups")
    if isinstance(required_lookups, list):
        stats["num_required_lookups"] = len(required_lookups)
    return stats


def read_realized_scale(settings_path: str) -> dict:
    """`calibrate_settings(..., scales=[scale])` sonrası settings.json'dan
    GERÇEKLEŞEN `input_scale`/`param_scale`/`logrows`'u okur (istenenle
    aynı olması BEKLENİR ama garanti değildir — ikisi de ayrı ayrı
    kaydediliyor). `run_args` beklenen şekilde değilse boş dict + UYARI."""
    with open(settings_path, encoding="utf-8") as f:
        settings_json = json.load(f)
    run_args = settings_json.get("run_args")
    if not isinstance(run_args, dict):
        print(f"[bench_circuit] UYARI: settings.json'da beklenen 'run_args' dict'i yok (anahtarlar: {list(settings_json.keys())}).")
        return {}
    return {"input_scale": run_args.get("input_scale"), "param_scale": run_args.get("param_scale"), "logrows": run_args.get("logrows")}


def classify_exception_status(e: BaseException) -> str:
    """ezkl'nin Rust tarafı bazı kombinasyonlarda gerçek bir Rust panic'i
    `pyo3_runtime.PanicException` olarak fırlatabiliyor (BaseException'dan
    türer, `except Exception` YAKALAMAZ — Colab'da scale=16'da gözlendi).
    Bunu sıradan bir "failed"den AYRI "panic" olarak işaretliyoruz —
    hangi kombinasyonların ezkl'i gerçekten çökerttiği kaybolmasın."""
    if type(e).__name__ == "PanicException":
        return "panic"
    return "failed"


def build_multi_input_json(z: torch.Tensor, c: torch.Tensor) -> dict:
    flat_z = z.detach().cpu().numpy().reshape(-1).tolist()
    flat_c = c.detach().cpu().numpy().reshape(-1).tolist()
    return {"input_data": [flat_z, flat_c]}


def write_json_file(data: dict, path: str) -> None:
    assert_writable(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_bench_results(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_bench_results(results: dict, path: str) -> None:
    assert_writable(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    assert_writable(tmp_path)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def measure_peak_rss_kb() -> int | None:
    """Çağıran sürecin yaşam-boyu tepe RSS'i (KB, Linux). `resource` yoksa
    (Windows) None döner — uydurmaz."""
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def default_work_root() -> str:
    """Kombinasyon başına ezkl ara dosyalarının (pk/vk/derlenmiş devre/witness)
    yazılacağı varsayılan kök. `/content` varsa (Colab) ORAYA — VM'nin yerel
    diski, Drive DEĞİL, hem daha hızlı hem kota yemiyor. Yoksa (yerel/Windows,
    zaten ezkl çalışmıyor) sistem temp dizini — sadece `--help`/saf-fonksiyon
    testlerinin bir varsayılan değere ihtiyacı var, path'in var olması gerekmez."""
    if os.path.isdir("/content"):
        return "/content/zk_bench_work"
    return os.path.join(tempfile.gettempdir(), "zk_bench_work")


def compute_dir_size_bytes(path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                pass
    return total


def cleanup_work_dir(work_dir: Path, keep_artifacts: bool) -> dict:
    """`work_dir`'deki BÜYÜK ara dosyaları (`CLEANUP_DELETE_FILENAMES`) siler
    — boyutları zaten `result.json`'a kaydedildiğinden dosyanın kendisine
    ihtiyaç yok. `settings.json`/`proof.json`/`Verifier.sol`/`.abi`/`vk.key`/
    `input.json` DOKUNULMAZ (silme listesinde YOK, örtük olarak korunur).
    `keep_artifacts=True` ise HİÇBİR ŞEY silinmez (elle inceleme için)."""
    before = compute_dir_size_bytes(work_dir)
    deleted_files: list[str] = []
    if not keep_artifacts:
        for name in CLEANUP_DELETE_FILENAMES:
            file_path = Path(work_dir) / name
            if file_path.is_file():
                size = file_path.stat().st_size
                file_path.unlink()
                deleted_files.append(name)
                print(f"[bench_circuit]  Silindi: {file_path} ({size / 1e6:.1f} MB)")
    after = compute_dir_size_bytes(work_dir)
    print(
        f"[bench_circuit]  Disk kullanımı ({work_dir}): {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB "
        f"(silinen: {(before - after) / 1e6:.1f} MB, keep_artifacts={keep_artifacts})"
    )
    return {
        "disk_usage_before_cleanup_bytes": before,
        "disk_usage_after_cleanup_bytes": after,
        "disk_usage_freed_bytes": before - after,
        "cleaned_up_files": deleted_files,
        "keep_artifacts": keep_artifacts,
    }


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "evet" if value else "hayır"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown_table(results: dict) -> str:
    headers = [
        "kombinasyon", "durum", "gerceklesen_scale", "gen_settings+calibrate(s)", "compile(s)", "setup(s)",
        "gen_witness(s)", "prove(s)", "verify_offchain(s)", "proof(B)", "pk(B)", "vk(B)",
        "tepe_RAM(KB)", "decomposition_uyari", "max_abs_error", "max_abs_error_%", "disk_oncesi(MB)", "disk_sonrasi(MB)",
        "deployed_bytecode(B)", "EIP170_asiyor", "verify_gas", "hata",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for key, r in results.items():
        timings = r.get("timings") or {}
        gen_calib = None
        if "gen_settings" in timings or "calibrate_settings" in timings:
            gen_calib = timings.get("gen_settings", 0.0) + timings.get("calibrate_settings", 0.0)
        realized_scale = r.get("realized_scale") or {}
        row = [
            key,
            r.get("status", "?"),
            _cell(realized_scale.get("input_scale") if isinstance(realized_scale, dict) else realized_scale),
            _cell(gen_calib),
            _cell(timings.get("compile_circuit")),
            _cell(timings.get("setup")),
            _cell(timings.get("gen_witness")),
            _cell(timings.get("prove")),
            _cell(timings.get("verify_offchain")),
            _cell(r.get("proof_size_bytes")),
            _cell(r.get("pk_size_bytes")),
            _cell(r.get("vk_size_bytes")),
            _cell(r.get("peak_rss_kb")),
            _cell(r.get("decomposition_warning_count")),
            _cell(r.get("max_abs_error")),
            _cell(r.get("max_abs_error_relative_pct")),
            _cell(round(r["disk_usage_before_cleanup_bytes"] / 1e6, 1) if r.get("disk_usage_before_cleanup_bytes") is not None else None),
            _cell(round(r["disk_usage_after_cleanup_bytes"] / 1e6, 1) if r.get("disk_usage_after_cleanup_bytes") is not None else None),
            _cell(r.get("deployed_bytecode_size")),
            _cell(r.get("exceeds_eip170")),
            _cell(r.get("verify_gas")),
            _cell(r.get("error_summary")),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ezkl/zincir'e bağımlı adımlar (BİLİNÇLİ TEST SINIRI — sadece Colab)
# ---------------------------------------------------------------------------


def run_ezkl_pipeline(onnx_path: Path, input_json_path: Path, work_dir: Path, scale: int) -> dict:
    import ezkl

    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "settings": work_dir / "settings.json",
        "compiled": work_dir / "network.compiled",
        "pk": work_dir / "pk.key",
        "vk": work_dir / "vk.key",
    }
    timings: dict = {}

    # NOT: input_scale/param_scale BURADA verilmiyor (gen_settings zamanında) —
    # diagnose_calibration.py'nin Colab bulgusuna göre bu yöntem (+ calibrate
    # sonrası settings.json'u elle yamama, force_settings_scale) HİÇBİR ölçekte
    # tutmadı. Çalışan yöntem: calibrate_settings'e scales=[scale] vermek.
    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = INPUT_VISIBILITY
    run_args.param_visibility = PARAM_VISIBILITY
    run_args.output_visibility = OUTPUT_VISIBILITY

    print("[bench_circuit]  gen_settings (scale henüz belirlenmedi, calibrate belirleyecek)...")
    t0 = time.perf_counter()
    ok = run_async(ezkl.gen_settings, str(onnx_path), str(paths["settings"]), py_run_args=run_args)
    if ok is not True:
        raise RuntimeError(f"gen_settings True döndürmedi: {ok!r}")
    timings["gen_settings"] = time.perf_counter() - t0

    print(f"[bench_circuit]  calibrate_settings (target=resources, scales=[{scale}])...")
    t0 = time.perf_counter()
    run_async(ezkl.calibrate_settings, str(input_json_path), str(onnx_path), str(paths["settings"]), "resources", scales=[scale])
    timings["calibrate_settings"] = time.perf_counter() - t0

    realized_scale = read_realized_scale(str(paths["settings"]))
    print(f"[bench_circuit]  istenen scale={scale} -> gerçekleşen: {realized_scale}")
    if realized_scale.get("input_scale") != scale:
        print(f"[bench_circuit]  UYARI: gerçekleşen input_scale ({realized_scale.get('input_scale')}) istenenden ({scale}) FARKLI.")

    print("[bench_circuit]  compile_circuit...")
    t0 = time.perf_counter()
    ok = run_async(ezkl.compile_circuit, str(onnx_path), str(paths["compiled"]), str(paths["settings"]))
    if ok is not True:
        raise RuntimeError(f"compile_circuit True döndürmedi: {ok!r}")
    timings["compile_circuit"] = time.perf_counter() - t0

    with open(paths["settings"], encoding="utf-8") as f:
        final_settings = json.load(f)
    circuit_stats = extract_circuit_stats(final_settings)
    print(f"[bench_circuit]  Devre istatistikleri (best-effort): {circuit_stats}")

    print("[bench_circuit]  get_srs...")
    t0 = time.perf_counter()
    run_get_srs(str(paths["settings"]))
    timings["get_srs"] = time.perf_counter() - t0

    print("[bench_circuit]  setup (pk/vk üretiliyor)...")
    t0 = time.perf_counter()
    ok = run_async(ezkl.setup, str(paths["compiled"]), str(paths["vk"]), str(paths["pk"]))
    if ok is not True:
        raise RuntimeError(f"setup True döndürmedi: {ok!r}")
    timings["setup"] = time.perf_counter() - t0

    for name, p in paths.items():
        if not p.exists():
            raise RuntimeError(f"setup sonrası beklenen '{name}' dosyası yok: {p}")

    return {
        "paths": paths,
        "timings": timings,
        "circuit_stats": circuit_stats,
        "requested_scale": scale,
        "realized_scale": realized_scale,
        "pk_size_bytes": paths["pk"].stat().st_size,
        "vk_size_bytes": paths["vk"].stat().st_size,
    }


def run_prove_and_verify(input_json_path: Path, ezkl_paths: dict, work_dir: Path) -> dict:
    import ezkl

    witness_path = work_dir / "witness.json"
    proof_path = work_dir / "proof.json"
    timings: dict = {}

    print("[bench_circuit]  gen_witness...")
    t0 = time.perf_counter()
    run_async(ezkl.gen_witness, str(input_json_path), str(ezkl_paths["compiled"]), str(witness_path))
    if not witness_path.exists():
        raise RuntimeError(f"gen_witness sonrası witness dosyası yok: {witness_path}")
    timings["gen_witness"] = time.perf_counter() - t0

    print("[bench_circuit]  prove...")
    t0 = time.perf_counter()
    run_async(ezkl.prove, str(witness_path), str(ezkl_paths["compiled"]), str(ezkl_paths["pk"]), str(proof_path))
    if not proof_path.exists():
        raise RuntimeError(f"prove sonrası proof dosyası yok: {proof_path}")
    timings["prove"] = time.perf_counter() - t0

    print("[bench_circuit]  verify (offchain)...")
    t0 = time.perf_counter()
    verified = bool(run_async(ezkl.verify, str(proof_path), str(ezkl_paths["settings"]), str(ezkl_paths["vk"])))
    timings["verify_offchain"] = time.perf_counter() - t0
    if not verified:
        raise RuntimeError("ezkl.verify False döndürdü (offchain doğrulama başarısız).")

    return {
        "witness_path": witness_path,
        "proof_path": proof_path,
        "timings": timings,
        "proof_size_bytes": proof_path.stat().st_size,
    }


def run_worker(args: argparse.Namespace) -> int:
    """TEK bir (k, embed_mode, scale) kombinasyonunu uçtan uca çalıştırır.
    Ebeveyn süreç bunu `--worker` bayrağıyla bir alt süreç olarak
    başlatır (bkz. modül docstring'i — izolasyon gerekçeleri). Sonuç,
    başarılı da olsa olmasa da `--result-out`'a yazılır (finally
    bloğunda) — ebeveyn bu dosyayı okuyarak neyin nereye kadar gittiğini
    öğrenir."""
    from chain.anvil import AnvilProcess
    from circuits.toy_pipeline import (
        compile_verifier_solidity,
        deploy_and_verify_onchain,
        deploy_and_verify_via_ezkl_native,
        generate_solidity_verifier,
    )

    combo = {"k": args.k, "embed_mode": args.embed_mode, "scale": args.scale}
    key = combo_key(combo)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"combo": combo, "status": "failed", "error_summary": None, "timings": {}}
    t_start = time.perf_counter()

    try:
        onnx_path = Path(args.onnx_dir) / f"mapping_k{args.k}_{args.embed_mode}.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"ONNX dosyası yok: {onnx_path} (önce 'python -m circuits.export_mapping' çalıştırılmalı)"
            )

        reference = load_reference(args.reference_dir, args.k)
        z, c = reference["z"], reference["c"]
        # w'nin gerçek büyüklüğü (referans, num_ws=0 dilimi) — ebeveyn, alt sürecin
        # stdout'undan ayrıştırdığı max_abs_error'u buna göre ORANLAYIP (%) fidelity/
        # scale ödünleşimini bağlamsallaştırıyor (bkz. run_combo_in_subprocess).
        result["w_abs_max"] = reference["w"][:, 0, :].abs().max().item()

        input_json_path = work_dir / "input.json"
        write_json_file(build_multi_input_json(z, c), str(input_json_path))
        del reference
        gc.collect()

        setup_result = run_ezkl_pipeline(onnx_path, input_json_path, work_dir, args.scale)
        result["timings"].update(setup_result["timings"])
        result["circuit_stats"] = setup_result["circuit_stats"]
        result["pk_size_bytes"] = setup_result["pk_size_bytes"]
        result["vk_size_bytes"] = setup_result["vk_size_bytes"]
        ezkl_paths = setup_result["paths"]
        gc.collect()

        prove_result = run_prove_and_verify(input_json_path, ezkl_paths, work_dir)
        result["timings"].update(prove_result["timings"])
        result["proof_size_bytes"] = prove_result["proof_size_bytes"]
        gc.collect()

        sol_path, _abi_path = generate_solidity_verifier(ezkl_paths, work_dir)

        t0 = time.perf_counter()
        try:
            compiled = compile_verifier_solidity(sol_path)
            result["timings"]["solc_compile"] = time.perf_counter() - t0
            result["deployed_bytecode_size"] = compiled["deployed_bytecode_size"]
            result["exceeds_eip170"] = compiled["exceeds_eip170"]
            result["used_solc_strategy"] = compiled.get("used_strategy")

            with AnvilProcess() as anvil:
                onchain = deploy_and_verify_onchain(compiled["bytecode"], prove_result["proof_path"], work_dir, anvil)
        except RuntimeError as e:
            print(f"[bench_circuit] Tüm solc stratejileri başarısız, ezkl native deploy'a düşülüyor: {e}")
            result["solc_fallback_reason"] = str(e)
            with AnvilProcess() as anvil:
                onchain = deploy_and_verify_via_ezkl_native(sol_path, prove_result["proof_path"], work_dir, anvil)

        result["contract_address"] = onchain["contract_address"]
        result["deploy_gas"] = onchain["deploy_gas"]
        result["verify_gas"] = onchain["verify_gas"]
        result["verified_onchain"] = onchain["verified"]

        result["status"] = "success"

    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa: BLE001 - pyo3 PanicException DAHIL tüm hatalar burada yakalanıp raporlanıyor, koşu düşmüyor
        result["status"] = classify_exception_status(e)
        result["error_summary"] = f"{type(e).__module__}.{type(e).__name__}: {e}"
        print(f"[bench_circuit] KOMBİNASYON {result['status'].upper()} ({key}): {result['error_summary']}")
        print(traceback.format_exc())

    finally:
        result["total_wall_seconds"] = time.perf_counter() - t_start
        result["peak_rss_kb"] = measure_peak_rss_kb()
        # Boyutlar (pk/vk/proof) yukarıda zaten result'a kaydedildi - dosyaların
        # kendisine artık ihtiyaç yok, büyük olanları burada temizliyoruz.
        result.update(cleanup_work_dir(work_dir, keep_artifacts=args.keep_artifacts))
        write_json_file(result, args.result_out)
        print(f"[bench_circuit] Sonuç yazıldı: {args.result_out}")
        gc.collect()

    return 0 if result["status"] == "success" else 1


def run_combo_in_subprocess(
    combo: dict, onnx_dir: str, reference_dir: str, work_root: Path, results_root: Path, timeout: float, keep_artifacts: bool
) -> dict:
    key = combo_key(combo)
    combo_work_dir = work_root / key
    combo_work_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_root / f"{key}.json"
    log_path = results_root / f"{key}.log.txt"

    cmd = [
        sys.executable, "-m", "scripts.bench_circuit", "--worker",
        "--k", str(combo["k"]), "--embed-mode", combo["embed_mode"], "--scale", str(combo["scale"]),
        "--onnx-dir", onnx_dir, "--reference-dir", reference_dir,
        "--work-dir", str(combo_work_dir), "--result-out", str(result_path),
    ]
    if keep_artifacts:
        cmd.append("--keep-artifacts")
    print(f"\n=== Kombinasyon: {key} (timeout={timeout}s) ===")
    print(f"[bench_circuit] Komut: {' '.join(cmd)}")

    popen_kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True  # anvil gibi torun süreçleri de grup halinde öldürebilmek için
    proc = subprocess.Popen(cmd, **popen_kwargs)

    timed_out = False
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(f"[bench_circuit] '{key}' {timeout}s içinde bitmedi, sonlandırılıyor.")
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        stdout, _ = proc.communicate()

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(stdout or "")

    if result_path.is_file():
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = {
            "combo": combo,
            "status": "failed",
            "error_summary": "Worker süreci sonuç dosyası yazmadan sonlandı (çökme/timeout-kill).",
            "timings": {},
            "peak_rss_kb": None,
        }

    if timed_out:
        result["status"] = "failed"
        result["error_summary"] = f"{result.get('error_summary') or ''} [TIMEOUT: {timeout}s aşıldı]".strip()

    result["returncode"] = proc.returncode
    result["decomposition_warning_count"] = count_decomposition_warnings(stdout or "")
    result["max_abs_error"] = extract_max_abs_error(stdout or "")
    result["log_path"] = str(log_path)

    # scale/fidelity ödünleşimi için: max_abs_error'u w'nin gerçek büyüklüğüne (worker'ın
    # kendi kaydettiği w_abs_max) oranlayıp yüzde olarak da raporluyoruz (makale tablosu).
    w_abs_max = result.get("w_abs_max")
    if result.get("max_abs_error") is not None and w_abs_max:
        result["max_abs_error_relative_pct"] = 100.0 * result["max_abs_error"] / w_abs_max
    else:
        result["max_abs_error_relative_pct"] = None

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz C3: ezkl devre benchmark ızgarası (k x embed_mode x scale).")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--onnx-dir", default=None, help="varsayılan: {zk_root}/circuits")
    parser.add_argument("--reference-dir", default=None, help="varsayılan: {zk_root}/reference")
    parser.add_argument("--bench-dir", default=None, help="varsayılan: {zk_root}/bench (Drive) — SADECE küçük sonuç dosyaları (result json'ları, bench_results.json)")
    parser.add_argument(
        "--work-root", default=None,
        help="varsayılan: /content/zk_bench_work (Colab, yerel disk) ya da sistem temp — BÜYÜK ezkl ara dosyaları (pk/vk/witness/settings) buraya",
    )
    parser.add_argument("--keep-artifacts", action="store_true", help="büyük ara dosyaları (pk/compiled/witness) silme, work_root'ta bırak")
    parser.add_argument("--markdown-out", default="docs/phase_c_bench.md")
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--embed-modes", default=",".join(DEFAULT_EMBED_MODES))
    parser.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    parser.add_argument("--only-first", action="store_true", help="sadece ilk (en küçük) kombinasyonu koş")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="kombinasyon başına saniye")
    parser.add_argument("--force", action="store_true", help="zaten sonuçlandırılmış kombinasyonları yeniden çalıştır")
    # Aşağıdakiler SADECE alt süreç (--worker) tarafından kullanılır; kullanıcıya gösterilmez.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--k", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--embed-mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scale", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-out", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.worker:
        return run_worker(args)

    paths = load_paths(env=args.env, config_path=args.paths_config)
    onnx_dir = args.onnx_dir or os.path.join(paths["zk_root"], "circuits")
    reference_dir = args.reference_dir or os.path.join(paths["zk_root"], "reference")
    bench_dir = Path(args.bench_dir or os.path.join(paths["zk_root"], "bench"))
    work_root = Path(args.work_root or default_work_root())

    k_values = parse_k_values(args.k_values)
    embed_modes = [m.strip() for m in args.embed_modes.split(",") if m.strip()]
    scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]

    print(f"[bench_circuit] onnx_dir={onnx_dir}")
    print(f"[bench_circuit] reference_dir={reference_dir}")
    print(f"[bench_circuit] bench_dir={bench_dir} (KÜÇÜK sonuç dosyaları — Drive)")
    print(f"[bench_circuit] work_root={work_root} (BÜYÜK ara dosyalar — yerel disk, --keep-artifacts yoksa her kombinasyon sonrası temizlenir)")
    print(f"[bench_circuit] k_values={k_values} embed_modes={embed_modes} scales={scales}")
    print(f"[bench_circuit] timeout={args.timeout}s only_first={args.only_first} force={args.force} keep_artifacts={args.keep_artifacts}")

    assert_writable(str(bench_dir))
    bench_dir.mkdir(parents=True, exist_ok=True)
    results_root = bench_dir / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    assert_writable(str(work_root))
    work_root.mkdir(parents=True, exist_ok=True)

    grid = build_grid(k_values, embed_modes, scales, only_first=args.only_first)
    print(f"[bench_circuit] Izgara: {len(grid)} kombinasyon")
    for combo in grid:
        print(f"  - {combo_key(combo)}")

    results_path = str(bench_dir / "bench_results.json")
    all_results = load_bench_results(results_path)

    for combo in grid:
        key = combo_key(combo)
        if key in all_results and not args.force:
            print(f"\n[bench_circuit] '{key}' zaten sonuçlandırılmış (durum={all_results[key].get('status')}), atlanıyor (--force ile yeniden çalıştır).")
            continue

        result = run_combo_in_subprocess(combo, onnx_dir, reference_dir, work_root, results_root, args.timeout, args.keep_artifacts)
        all_results[key] = result
        save_bench_results(all_results, results_path)

        print(
            f"[bench_circuit] '{key}' bitti: durum={result['status']} "
            f"decomposition_uyari={result.get('decomposition_warning_count')} "
            f"tepe_RAM_KB={result.get('peak_rss_kb')} "
            f"disk_sonrasi_MB={(result.get('disk_usage_after_cleanup_bytes') or 0) / 1e6:.1f}"
        )
        if result["status"] != "success":
            print(f"[bench_circuit]   hata: {result.get('error_summary')}")

        gc.collect()

    print("\n=== Faz C3 benchmark sonuçları ===")
    print(render_markdown_table(all_results))

    md_path = args.markdown_out
    assert_writable(md_path)
    os.makedirs(os.path.dirname(os.path.abspath(md_path)) or ".", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Faz C3 — Devre Benchmark Sonuçları\n\n{render_markdown_table(all_results)}")
    print(f"[bench_circuit] Yazıldı: {md_path}")
    print(f"[bench_circuit] Yazıldı: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
