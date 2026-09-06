"""fp32 FedAvg ortalaması, state_dict karşılaştırması ve norm istatistikleri.

Saf torch/numpy mantığıdır; StyleGAN'a bağımlı değildir. `RunningAverage`
çağıran tarafın büyük state_dict'leri (pkl'dan G_ema.state_dict()) tek
tek yükleyip `update` çağırmasına, ardından objeyi serbest bırakmasına
izin verir — aynı anda birden fazla tam generator belleğe alınmaz.
"""

from __future__ import annotations

import numpy as np
import torch


class RunningAverage:
    def __init__(self):
        self._sum: dict | None = None
        self._keys: set | None = None
        self._count = 0

    def update(self, state_dict: dict) -> None:
        keys = set(state_dict.keys())
        if self._sum is None:
            self._keys = keys
            self._sum = {k: v.detach().cpu().to(torch.float32).clone() for k, v in state_dict.items()}
            self._count = 1
            return

        if keys != self._keys:
            raise ValueError(
                f"Anahtar kümesi uyuşmuyor. Eksik: {self._keys - keys}, "
                f"fazladan: {keys - self._keys}"
            )

        for k in self._keys:
            self._sum[k] += state_dict[k].detach().cpu().to(torch.float32)
        self._count += 1

    def result(self) -> dict:
        if self._sum is None or self._count == 0:
            raise RuntimeError("result() çağrılmadan önce en az bir update() gerekli.")
        return {k: v / self._count for k, v in self._sum.items()}

    @property
    def count(self) -> int:
        return self._count


def compare_state_dicts(a: dict, b: dict) -> dict:
    keys_a, keys_b = set(a.keys()), set(b.keys())
    common = sorted(keys_a & keys_b)

    per_key = {}
    for key in common:
        ta = a[key].detach().cpu().to(torch.float32).flatten()
        tb = b[key].detach().cpu().to(torch.float32).flatten()
        if ta.shape != tb.shape:
            per_key[key] = {"error": f"şekil uyuşmuyor: {list(ta.shape)} vs {list(tb.shape)}"}
            continue

        max_abs_diff = torch.max(torch.abs(ta - tb)).item()
        denom = torch.linalg.norm(ta) * torch.linalg.norm(tb)
        cosine = (torch.dot(ta, tb) / denom).item() if denom > 0 else None
        per_key[key] = {"max_abs_diff": max_abs_diff, "cosine_similarity": cosine}

    overall_max_diff = max(
        (v["max_abs_diff"] for v in per_key.values() if "max_abs_diff" in v), default=None
    )

    return {
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
        "per_key": per_key,
        "overall_max_abs_diff": overall_max_diff,
    }


def compute_norm_percentiles(deltas: list[float]) -> dict:
    if not deltas:
        raise ValueError("Boş delta listesinden percentile hesaplanamaz.")
    arr = np.asarray(deltas, dtype=np.float64)
    p50, p90, p99 = np.percentile(arr, [50, 90, 99])
    return {"p50": float(p50), "p90": float(p90), "p99": float(p99), "n": len(deltas)}
