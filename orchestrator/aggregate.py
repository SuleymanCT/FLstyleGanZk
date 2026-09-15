"""Faz D: doğrulama-kapılı FedAvg.

`fl.fedavg_utils.RunningAverage`'i DOĞRUDAN kullanır (yeniden yazmaz) —
bu, Faz A'nın `scripts/audit_fedavg.py`'sinin doğruladığı fp32 ortalama
ile BİREBİR aynı aritmetiği garanti eder (aynı sınıf, iki ayrı yerde
çağrılıyor). Bu modülün eklediği şey saf aritmetik DEĞİL: hangi
site'lerin ortalamaya DAHİL edileceğine karar veren KAPI (norm/NaN-Inf/
zincir-onayı kontrolleri).

KURAL (CLAUDE.md madde 5): burada KUANTİZASYON YAPILMAZ. Kuantize kopya
sadece ispat için AYRI üretilir (bkz. `circuits/export_mapping.py`),
orijinal fp32 ağırlıkları bu modül asla değiştirmez.
"""

from __future__ import annotations

import math

import torch

from fl.fedavg_utils import RunningAverage


def compute_delta_norm(prev_state: dict, new_state: dict) -> float:
    """`||new - prev||_2` — TÜM tensörler tek bir vektör gibi
    düzleştirilip birleştirilir (toplam L2 norm, katman katman DEĞİL) —
    `tau_norm_threshold` (Faz A'nın gerçek p99 ölçümüne dayalı) bu
    toplam normla karşılaştırılıyor."""
    if set(prev_state.keys()) != set(new_state.keys()):
        raise ValueError("prev_state ve new_state anahtar kümeleri farklı.")
    total_sq = 0.0
    for key in prev_state:
        diff = new_state[key].detach().cpu().to(torch.float32) - prev_state[key].detach().cpu().to(torch.float32)
        total_sq += torch.sum(diff * diff).item()
    return math.sqrt(total_sq)


def has_nan_or_inf(state_dict: dict) -> bool:
    for tensor in state_dict.values():
        t = tensor.detach().cpu().to(torch.float32)
        if torch.isnan(t).any() or torch.isinf(t).any():
            return True
    return False


def gate_site_update(
    site: str, prev_global: dict, site_state: dict, *, tau_norm_threshold: float, chain_approved: bool
) -> dict:
    """Tek bir site güncellemesinin ortalamaya dahil EDİLİP EDİLMEYECEĞİNE
    karar verir. `chain_approved`: zincirdeki `submitProof`/itibar-eşiği
    sonucuna göre (round_runner.py belirler) bu site NE OLURSA OLSUN
    zincirce onaylı değilse dışlanır — yerel kontroller (norm/NaN-Inf)
    buna EK, tek başına zincir onayı yeterli DEĞİLDİR."""
    reasons: list[str] = []
    if not chain_approved:
        reasons.append("zincir onayı yok (ispat başarısız/istenmedi ya da itibar eşiği altı)")

    nan_inf = has_nan_or_inf(site_state)
    if nan_inf:
        reasons.append("NaN/Inf içeriyor")

    delta_norm = None
    if not nan_inf:
        delta_norm = compute_delta_norm(prev_global, site_state)
        if delta_norm > tau_norm_threshold:
            reasons.append(f"||ΔG||={delta_norm:.2f} > tau={tau_norm_threshold}")

    return {"site": site, "included": not reasons, "reasons": reasons, "delta_norm": delta_norm}


def aggregate_round(
    prev_global: dict, site_states: dict, *, tau_norm_threshold: float, chain_approved: dict
) -> dict:
    """Bir round'un TÜM site güncellemelerini kapılardan geçirip fp32
    FedAvg ile birleştirir. `chain_approved`: `{site: bool}` — zincirde
    ONAYLI olmayan site (round_runner.py'nin `RoundManagerClient.is_site_eligible`/
    submission durumuna göre doldurduğu) burada da dışlanır."""
    if not site_states:
        raise ValueError("site_states boş — agregasyon yapılamaz.")

    running = RunningAverage()
    gate_results = []
    included_sites = []

    for site, state in site_states.items():
        gate = gate_site_update(
            site, prev_global, state, tau_norm_threshold=tau_norm_threshold, chain_approved=chain_approved.get(site, False)
        )
        gate_results.append(gate)
        if gate["included"]:
            running.update(state)
            included_sites.append(site)
        else:
            print(f"[aggregate] '{site}' DIŞLANDI: {gate['reasons']}")

    if running.count == 0:
        raise RuntimeError("Hiçbir site ortalamaya dahil edilmedi — tüm site'ler dışlandı, agregasyon yapılamaz.")

    num_total = len(site_states)
    excluded_sites = [g["site"] for g in gate_results if not g["included"]]
    activation_rate = len(excluded_sites) / num_total
    if excluded_sites:
        print(
            f"[aggregate] UYARI: kapı(lar) devreye girdi — {len(excluded_sites)}/{num_total} site "
            f"dışlandı (aktivasyon oranı={activation_rate:.2%}): {excluded_sites}"
        )

    return {
        "global_state": running.result(),
        "included_sites": included_sites,
        "excluded_sites": excluded_sites,
        "gate_results": gate_results,
        "activation_rate": activation_rate,
        "num_included": running.count,
        "num_total": num_total,
    }
