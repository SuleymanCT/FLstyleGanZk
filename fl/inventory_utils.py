"""training_options.json / stats.jsonl / dosya yolu sınıflandırma yardımcıları.

Saf dict/string işlemleridir; hiçbir StyleGAN veya dosya sistemi
bağımlılığı içermez (dosya OKUMA işlemi çağıran tarafta yapılır, bu
modül sadece içerikle çalışır) — bu yüzden sentetik verilerle tam
test edilebilir.
"""

from __future__ import annotations

import json
import re

ROUND_SITE_PATTERN = re.compile(r"round_\d+[\\/]site_\d+")


def compare_training_options(options_by_path: dict[str, dict]) -> dict:
    all_keys = set()
    for opts in options_by_path.values():
        all_keys.update(opts.keys())

    differences = {}
    for key in sorted(all_keys):
        values_by_path = {}
        for path, opts in options_by_path.items():
            values_by_path[path] = opts.get(key, "<eksik>")

        distinct_values = {json.dumps(v, sort_keys=True, default=str) for v in values_by_path.values()}
        if len(distinct_values) > 1:
            differences[key] = values_by_path

    return {
        "num_files": len(options_by_path),
        "all_keys": sorted(all_keys),
        "differing_keys": differences,
    }


def parse_stats_jsonl(lines: list[str]) -> dict:
    series: dict[str, list] = {}
    warnings = []

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as e:
            warnings.append({"satir_no": line_no, "hata": str(e), "icerik": stripped[:200]})
            continue

        if not isinstance(record, dict):
            warnings.append({"satir_no": line_no, "hata": "JSON bir obje değil", "icerik": stripped[:200]})
            continue

        step = record.get("step", line_no)
        for key, value in record.items():
            if "fid" in key.lower():
                series.setdefault(key, []).append({"step": step, "value": value})

    return {"fid_series": series, "warnings": warnings, "num_lines": len(lines)}


def classify_pkl_paths(all_pkl_paths: list[str], raw_root: str) -> dict:
    expected = []
    base_model_candidates = []

    for path in all_pkl_paths:
        relative = path[len(raw_root):] if path.startswith(raw_root) else path
        if ROUND_SITE_PATTERN.search(relative):
            expected.append(path)
        else:
            base_model_candidates.append(path)

    return {
        "expected": sorted(expected),
        "base_model_candidates": sorted(base_model_candidates),
    }
