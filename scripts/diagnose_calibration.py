#!/usr/bin/env python
"""Faz C3 teşhis: ilk benchmark kombinasyonu (k=1, matmul, scale=8)
`calibrate_settings`'te düştü:

    RuntimeError: Failed to calibrate settings: [Uncategorized]
    calibration failed, could not find any suitable parameters given
    the calibration dataset

İki olası sebep var:
  (a) scale=8 gerçekten dar (z/c/w değer aralığı o scale'de temsil
      edilemiyor, ya da normalize_2nd_moment/leaky_relu gibi doğrusal
      olmayan işlemlerin lookup-tablosu aralığı yetersiz kalıyor).
  (b) çoklu-girdi `input.json` şeması yanlış — `scripts/bench_circuit.py`'de
      "BİLİNÇLİ VARSAYIM" olarak işaretlenmişti: `{"input_data": [flat_z, flat_c]}`.

## (b) için doğrulama — VARSAYIM DEĞİL, ezkl'in KENDİ kaynağından

`ezkl==23.0.5` (tam bu pinlenmiş etiket — `v23.0.5` — WebFetch ile
GERÇEKTEN çekildi, ana branch DEĞİL, versiyon kayması riski yok):

- `src/graph/input.rs`: `pub type DataSource = Vec<Vec<FileSourceInner>>`.
  Yani `input_data`'nın DIŞ listesinin her elemanı GRAFİĞİN AYRI BİR
  GİRDİSİNE karşılık gelir, İÇ liste o girdinin DÜZLEŞTİRİLMİŞ
  değerleridir. Bizim `{"input_data": [flat_z, flat_c]}` şemamız
  (`circuits.export_mapping`'teki `torch.onnx.export(..., input_names=["z","c"])`
  sırasıyla BİREBİR eşleşiyor) bu tanıma UYUYOR — şema muhtemelen DOĞRU.
- `ezkl.pyi` (aynı `v23.0.5` etiketi): `calibrate_settings(data, model,
  settings, target, lookup_safety_margin, scales, scale_rebase_multiplier,
  max_logrows)` — yani `scales: Optional[Sequence[int]]` ve
  `max_logrows: Optional[int]` GERÇEKTEN VAR ve pinlenmiş sürümümüzde
  mevcut. `circuits/bench_circuit.py`'nin şu anki `force_settings_scale`
  yaması (calibrate SONRASI settings.json'u elle düzeltmek) bu yüzden
  muhtemelen GEREKSİZ — doğrudan `scales=[scale]` verilebilir. Bu script
  ADIM 3'te ikisini de dener.

Kaynak kodu okuması "muhtemelen doğru" diyor ama GERÇEK ÇALIŞMA
DAVRANIŞI (pyo3 bağlamasının bu imzayı gerçekten kwarg ile kabul ettiği,
`input_data`'nın gerçekten varsayıldığı gibi yorumlandığı) ancak Colab'da
bu script çalıştırılınca kesinleşir — bu yüzden ADIM 1 hâlâ şema/şekil/
değer aralığını GERÇEK verilerle tekrar gözle doğruluyor, körü körüne
güvenmiyor.

BİLİNÇLİ TEST SINIRI: ADIM 1c/2/3/4 (gen_settings/calibrate_settings/
compile_circuit/setup/gen_witness/prove) `ezkl` gerektirir, SADECE
Colab'da çalışır — bu yüzden `ezkl` modül İÇİNDE (fonksiyon gövdesinde)
import edilir, script'in kendisi (`--help` dahil) ve ADIM 1/1b saf
kısımları yerelde çalışır/test edilir (`tests/test_diagnose_calibration.py`).

Kullanım (Colab'da):
    python -m scripts.diagnose_calibration --env colab --k 1 --embed-mode matmul
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import onnx
import torch

from circuits.ezkl_utils import run_async, run_get_srs
from circuits.export_mapping import load_reference
from circuits.rebuild_mapping import VALID_EMBED_MODES
from configs.loader import load_paths
from scripts.bench_circuit import build_multi_input_json, write_json_file
from storage.pathguard import assert_writable

_SCHEMA_FINDINGS_TEXT = """\
[ezkl==23.0.5 (v23.0.5 etiketi) kaynak/tip-stub okuması — WebFetch ile GERÇEKTEN çekildi]

1. src/graph/input.rs:
     pub type DataSource = Vec<Vec<FileSourceInner>>;
     pub struct GraphData { pub input_data: DataSource, ... }
   -> DIŞ liste = graf GİRDİSİ başına bir eleman, İÇ liste = o girdinin
      düzleştirilmiş değerleri. Bizim şemamız {"input_data": [flat_z, flat_c]}
      (input_names=["z","c"] sırasıyla) bu tanıma uyuyor.

2. ezkl.pyi:
     def calibrate_settings(data, model, settings, target,
         lookup_safety_margin, scales, scale_rebase_multiplier, max_logrows) -> Any
   -> 'scales: Optional[Sequence[int]]' ve 'max_logrows: Optional[int]'
      GERÇEKTEN var, tam bu pinlenmiş sürümde (versiyon kayması riski yok).

Bu ikisi "şema muhtemelen doğru" + "scale'i zorlamanın DAHA DOĞRU yolu
scales=[scale] kwarg'ı" diyor — ama gerçek pyo3 çalışma davranışı ancak
aşağıdaki adımlarla (Colab'da) kesinleşir.
"""


# ---------------------------------------------------------------------------
# Saf/testable yardımcılar (ezkl GEREKTİRMEZ)
# ---------------------------------------------------------------------------


def describe_tensor_stats(t: torch.Tensor) -> dict:
    flat = t.detach().to(torch.float32).flatten()
    return {
        "shape": list(t.shape),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "mean": flat.mean().item(),
        "abs_max": flat.abs().max().item(),
    }


def describe_input_json_shape(payload: dict) -> dict:
    input_data = payload.get("input_data", [])
    return {"num_inputs": len(input_data), "lengths": [len(entry) for entry in input_data]}


def read_onnx_input_shapes(onnx_path: str) -> dict:
    """ONNX grafiğinin DECLARE ettiği girdi şekillerini okur — ezkl de
    AYNI dosyayı okuyacağından, bu bizim varsaydığımız şeklin ezkl'in
    göreceğiyle tutarlı olup olmadığını bağımsız doğrular."""
    model = onnx.load(str(onnx_path))
    shapes: dict = {}
    for inp in model.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
        shapes[inp.name] = dims
    return shapes


def cross_check_input_shapes(onnx_shapes: dict, json_lengths: list[int]) -> list[str]:
    """ONNX'in beklediği (sayısal) eleman sayısı ile input.json'daki her
    girdinin düzleştirilmiş uzunluğunu karşılaştırır. Uyuşmazlık varsa
    açıklayıcı bir uyarı listesi döner (boşsa uyumlu demektir)."""
    warnings: list[str] = []
    names = list(onnx_shapes.keys())
    if len(names) != len(json_lengths):
        warnings.append(f"ONNX girdi sayısı ({len(names)}) input.json girdi sayısıyla ({len(json_lengths)}) UYUŞMUYOR.")
        return warnings
    for name, length in zip(names, json_lengths):
        dims = onnx_shapes[name]
        numeric_dims = [d for d in dims if isinstance(d, int)]
        if not numeric_dims:
            continue
        expected = 1
        for d in numeric_dims:
            expected *= d
        if expected != length:
            warnings.append(f"'{name}': ONNX şekli {dims} (sayısal çarpım={expected}) ama input.json uzunluğu {length} — UYUŞMUYOR.")
    return warnings


def parse_max_logrows_values(raw: str) -> list:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(None if token.lower() == "none" else int(token))
    return values


def print_calibration_table(results: list) -> None:
    print(f"{'yontem':<16}{'target':<12}{'scale':<7}{'max_logrows':<13}{'durum':<10}{'sure(s)':<9}hata")
    for r in results:
        print(
            f"{r['method']:<16}{r['target']:<12}{r['scale']:<7}{str(r['max_logrows']):<13}"
            f"{r['status']:<10}{r['elapsed_seconds']:<9.2f}{r.get('error') or ''}"
        )


def print_skip_calibration_table(results: list) -> None:
    print(f"{'scale':<7}{'durum':<10}{'son_adim':<16}hata")
    for r in results:
        print(f"{r['scale']:<7}{r['status']:<10}{str(r['last_step']):<16}{r.get('error') or ''}")


# ---------------------------------------------------------------------------
# ezkl'e bağımlı adımlar (BİLİNÇLİ TEST SINIRI — sadece Colab)
# ---------------------------------------------------------------------------


def dump_full_settings(onnx_path: str, settings_path: str, scale: int) -> dict:
    import ezkl

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "private"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "public"
    run_args.input_scale = scale
    run_args.param_scale = scale

    ok = run_async(ezkl.gen_settings, str(onnx_path), str(settings_path), py_run_args=run_args)
    if ok is not True:
        raise RuntimeError(f"gen_settings True dönmedi: {ok!r}")

    with open(settings_path, encoding="utf-8") as f:
        settings = json.load(f)
    print(f"[diagnose] settings.json (scale={scale}) TAM içerik:")
    print(json.dumps(settings, indent=2))
    return settings


def attempt_calibration(onnx_path: Path, input_json_path: Path, work_dir: Path, scale: int, target: str, method: str, max_logrows=None) -> dict:
    """`method`: 'run_args_scale' (PyRunArgs.input_scale/param_scale ile,
    calibrate_settings'e scale VERİLMEDEN — mevcut bench_circuit.py
    davranışı) ya da 'scales_kwarg' (PyRunArgs'a scale VERİLMEDEN,
    calibrate_settings(..., scales=[scale]) ile — ezkl.pyi'de doğrulanan
    gerçek parametre)."""
    import ezkl

    tag = f"{method}_{target}_scale{scale}_maxlogrows{max_logrows}"
    settings_path = work_dir / f"settings_{tag}.json"

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "private"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "public"
    if method == "run_args_scale":
        run_args.input_scale = scale
        run_args.param_scale = scale
    elif method != "scales_kwarg":
        raise ValueError(f"Bilinmeyen method: {method!r}")

    attempt: dict = {"method": method, "target": target, "scale": scale, "max_logrows": max_logrows, "status": "failed", "error": None}
    t0 = time.perf_counter()
    try:
        ok = run_async(ezkl.gen_settings, str(onnx_path), str(settings_path), py_run_args=run_args)
        if ok is not True:
            raise RuntimeError(f"gen_settings True dönmedi: {ok!r}")

        calib_kwargs: dict = {}
        if method == "scales_kwarg":
            calib_kwargs["scales"] = [scale]
        if max_logrows is not None:
            calib_kwargs["max_logrows"] = max_logrows

        run_async(ezkl.calibrate_settings, str(input_json_path), str(onnx_path), str(settings_path), target, **calib_kwargs)
        attempt["status"] = "success"
    except Exception as e:  # noqa: BLE001 - teşhis amaçlı, her deneme ayrı raporlanıp devam ediliyor
        attempt["error"] = f"{type(e).__name__}: {e}"
        print(
            f"[diagnose] DENEME BAŞARISIZ (method={method}, target={target}, scale={scale}, "
            f"max_logrows={max_logrows}): {attempt['error']}",
            flush=True,
        )
        print(traceback.format_exc(), flush=True)
    attempt["elapsed_seconds"] = time.perf_counter() - t0

    if attempt["status"] == "success":
        with open(settings_path, encoding="utf-8") as f:
            final_settings = json.load(f)
        run_args_out = final_settings.get("run_args", {}) if isinstance(final_settings, dict) else {}
        attempt["resulting_scale"] = run_args_out.get("input_scale")
        attempt["resulting_logrows"] = run_args_out.get("logrows")
        print(
            f"[diagnose] DENEME BAŞARILI (method={method}, target={target}, scale={scale}, "
            f"max_logrows={max_logrows}) -> gerçekleşen input_scale={attempt['resulting_scale']}, "
            f"logrows={attempt['resulting_logrows']}",
            flush=True,
        )

    return attempt


def attempt_skip_calibration(onnx_path: Path, input_json_path: Path, work_dir: Path, scale: int) -> dict:
    """Kalibrasyonu tamamen ATLAYIP sabit scale ile compile->setup->
    gen_witness->prove dener. Hangi adımda (varsa) düştüğünü raporlar."""
    import ezkl

    tag = f"skip_calib_scale{scale}"
    settings_path = work_dir / f"settings_{tag}.json"
    compiled_path = work_dir / f"compiled_{tag}.ezkl"
    pk_path = work_dir / f"pk_{tag}.key"
    vk_path = work_dir / f"vk_{tag}.key"
    witness_path = work_dir / f"witness_{tag}.json"
    proof_path = work_dir / f"proof_{tag}.json"

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "private"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "public"
    run_args.input_scale = scale
    run_args.param_scale = scale

    result: dict = {"scale": scale, "status": "failed", "last_step": None, "error": None, "timings": {}}
    try:
        t0 = time.perf_counter()
        ok = run_async(ezkl.gen_settings, str(onnx_path), str(settings_path), py_run_args=run_args)
        if ok is not True:
            raise RuntimeError(f"gen_settings True dönmedi: {ok!r}")
        result["timings"]["gen_settings"] = time.perf_counter() - t0
        result["last_step"] = "gen_settings"

        t0 = time.perf_counter()
        ok = run_async(ezkl.compile_circuit, str(onnx_path), str(compiled_path), str(settings_path))
        if ok is not True:
            raise RuntimeError(f"compile_circuit True dönmedi: {ok!r}")
        result["timings"]["compile_circuit"] = time.perf_counter() - t0
        result["last_step"] = "compile_circuit"

        t0 = time.perf_counter()
        run_get_srs(str(settings_path))
        result["timings"]["get_srs"] = time.perf_counter() - t0
        result["last_step"] = "get_srs"

        t0 = time.perf_counter()
        ok = run_async(ezkl.setup, str(compiled_path), str(vk_path), str(pk_path))
        if ok is not True:
            raise RuntimeError(f"setup True dönmedi: {ok!r}")
        result["timings"]["setup"] = time.perf_counter() - t0
        result["last_step"] = "setup"

        t0 = time.perf_counter()
        run_async(ezkl.gen_witness, str(input_json_path), str(compiled_path), str(witness_path))
        if not witness_path.exists():
            raise RuntimeError("gen_witness sonrası witness dosyası yok.")
        result["timings"]["gen_witness"] = time.perf_counter() - t0
        result["last_step"] = "gen_witness"

        t0 = time.perf_counter()
        run_async(ezkl.prove, str(witness_path), str(compiled_path), str(pk_path), str(proof_path))
        if not proof_path.exists():
            raise RuntimeError("prove sonrası proof dosyası yok.")
        result["timings"]["prove"] = time.perf_counter() - t0
        result["last_step"] = "prove"

        result["status"] = "success"
        print(f"[diagnose] KALİBRASYONSUZ scale={scale}: TÜM ADIMLAR BAŞARILI ({result['timings']})", flush=True)
    except Exception as e:  # noqa: BLE001 - teşhis amaçlı, her scale ayrı raporlanıp devam ediliyor
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"[diagnose] KALİBRASYONSUZ scale={scale} '{result['last_step']}' adımından SONRA BAŞARISIZ: {result['error']}", flush=True)
        print(traceback.format_exc(), flush=True)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz C3 teşhis: kalibrasyon neden düşüyor (scale mi, input.json şeması mı)?")
    parser.add_argument("--env", default=None, choices=["local", "colab"])
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--onnx-dir", default=None, help="varsayılan: {zk_root}/circuits")
    parser.add_argument("--reference-dir", default=None, help="varsayılan: {zk_root}/reference")
    parser.add_argument("--work-dir", default=None, help="varsayılan: {zk_root}/bench/diagnose")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--embed-mode", default="matmul", choices=list(VALID_EMBED_MODES))
    parser.add_argument("--scales", default="8,11,13,16")
    parser.add_argument("--targets", default="resources,accuracy")
    parser.add_argument("--max-logrows-values", default="none,15,19,22")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = load_paths(env=args.env, config_path=args.paths_config)
    onnx_dir = args.onnx_dir or os.path.join(paths["zk_root"], "circuits")
    reference_dir = args.reference_dir or os.path.join(paths["zk_root"], "reference")
    work_dir = Path(args.work_dir or os.path.join(paths["zk_root"], "bench", "diagnose"))
    assert_writable(str(work_dir))
    work_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = Path(onnx_dir) / f"mapping_k{args.k}_{args.embed_mode}.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX dosyası yok: {onnx_path} (önce circuits.export_mapping çalıştırılmalı)")

    scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    max_logrows_values = parse_max_logrows_values(args.max_logrows_values)

    print(f"[diagnose] onnx_path={onnx_path}")
    print(f"[diagnose] reference_dir={reference_dir} k={args.k} embed_mode={args.embed_mode}")
    print(f"[diagnose] scales={scales} targets={targets} max_logrows_values={max_logrows_values}")

    print("\n" + "=" * 70)
    print("ADIM 1: input.json şeması netleştirmesi (ezkl v23.0.5 kaynağından)")
    print("=" * 70)
    print(_SCHEMA_FINDINGS_TEXT)

    reference = load_reference(reference_dir, args.k)
    z, c = reference["z"], reference["c"]
    w_ref = reference["w"][:, 0, :]

    input_payload = build_multi_input_json(z, c)
    input_json_path = work_dir / "input.json"
    write_json_file(input_payload, str(input_json_path))
    print(f"\nKullanılan input.json (tam içerik, {input_json_path}):")
    print(json.dumps(input_payload, indent=2))

    shape_info = describe_input_json_shape(input_payload)
    print(f"\ninput_data şekli (ayrıştırılmış): {shape_info}")

    onnx_shapes = read_onnx_input_shapes(str(onnx_path))
    print(f"ONNX grafiğinin girdi şekilleri (ezkl AYNI dosyayı okuyacak): {onnx_shapes}")
    mismatches = cross_check_input_shapes(onnx_shapes, shape_info["lengths"])
    if mismatches:
        print("UYUMSUZLUK BULUNDU (sebep (b) DOĞRULANDI):")
        for m in mismatches:
            print(f"  - {m}")
    else:
        print("Şekiller UYUŞUYOR — input.json ONNX grafiğiyle tutarlı (sebep (b) burada elenmiş görünüyor).")

    print("\n" + "=" * 70)
    print("ADIM 1b: z / c / w gerçek değer istatistikleri")
    print("=" * 70)
    z_stats, c_stats, w_stats = describe_tensor_stats(z), describe_tensor_stats(c), describe_tensor_stats(w_ref)
    print(f"z: {z_stats}")
    print(f"c: {c_stats}")
    print(f"w (referans, k={args.k}, num_ws=0 dilimi): {w_stats}")
    print(
        "Not: kalibrasyon başarısızlığının (a) sebebi (dar scale) ile ilişkisi, "
        "normalize_2nd_moment (rsqrt) / leaky_relu gibi doğrusal olmayan işlemlerin "
        "gerektirdiği lookup-tablosu aralığının düşük scale'de yetersiz kalmasından "
        "olabilir (bkz. docs/phase_b_report.md'deki decomposition uyarıları) — bu KESIN "
        "DEĞİL, aşağıdaki ampirik tarama gösterecek."
    )

    print("\n" + "=" * 70)
    print("ADIM 1c: gen_settings ile TAM settings.json dökümü (baseline scale)")
    print("=" * 70)
    baseline_scale = scales[0]
    dump_full_settings(str(onnx_path), str(work_dir / "settings_baseline.json"), baseline_scale)

    print("\n" + "=" * 70)
    print("ADIM 2/3: kalibrasyon ızgarası (target x scale x yöntem, + max_logrows taraması)")
    print("=" * 70)
    calib_results = []
    for target in targets:
        for scale in scales:
            for method in ("run_args_scale", "scales_kwarg"):
                calib_results.append(attempt_calibration(onnx_path, input_json_path, work_dir, scale, target, method))

    top_scale = max(scales)
    for target in targets:
        for max_logrows in max_logrows_values:
            if max_logrows is None:
                continue
            calib_results.append(
                attempt_calibration(onnx_path, input_json_path, work_dir, top_scale, target, "scales_kwarg", max_logrows=max_logrows)
            )

    print("\n--- Kalibrasyon sonuç tablosu ---")
    print_calibration_table(calib_results)

    print("\n" + "=" * 70)
    print("ADIM 4: kalibrasyonu ATLAYIP sabit scale ile compile+setup+witness+prove")
    print("=" * 70)
    skip_results = [attempt_skip_calibration(onnx_path, input_json_path, work_dir, scale) for scale in scales]
    print("\n--- Kalibrasyonsuz sonuç tablosu ---")
    print_skip_calibration_table(skip_results)

    report = {
        "onnx_path": str(onnx_path),
        "input_json_shape": shape_info,
        "onnx_input_shapes": onnx_shapes,
        "shape_mismatches": mismatches,
        "z_stats": z_stats,
        "c_stats": c_stats,
        "w_stats": w_stats,
        "calibration_attempts": calib_results,
        "skip_calibration_attempts": skip_results,
    }
    report_path = work_dir / "diagnose_report.json"
    write_json_file(report, str(report_path))
    print(f"\n[diagnose] Tam rapor yazıldı: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
