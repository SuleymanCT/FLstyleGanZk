#!/usr/bin/env python
"""Tek seferlik keşif scripti: bir fedavg_*.pt dosyasının gerçek yapısı ne?

`scripts/audit_fedavg.py` fedavg_*.pt'nin düz {parametre_adı: tensör}
olduğunu varsayıyordu; gerçekte iç içe bir sözlük olduğu ortaya çıktı
(bkz. `AttributeError: 'dict' object has no attribute 'detach'`). Bu
script hiçbir varsayımda bulunmadan yapıyı olduğu gibi yazdırır —
düzeltme, buradan gelecek gerçek çıktı görüldükten sonra yapılacak.

BİLİNÇLİ TEST SINIRI: gerçek bir fedavg_*.pt dosyası ve Drive erişimi
gerektirir, yerelde test edilemez.

Kullanım (Colab'da):
    python -m scripts.probe_fedavg --path /content/drive/MyDrive/.../fedavg_0.pt
"""

from __future__ import annotations

import argparse

import torch

from storage.pathguard import open_readonly

DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_ITEMS = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="fedavg_*.pt dosyasının yapısını özyinelemeli yazdır.")
    parser.add_argument("--path", required=True, help="İncelenecek fedavg_*.pt dosyasının tam yolu.")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    return parser.parse_args(argv)


def describe_value(value, depth: int, max_depth: int, max_items: int, indent: str = "") -> None:
    if torch.is_tensor(value):
        print(f"{indent}tensör  şekil={tuple(value.shape)}  dtype={value.dtype}")
        return

    if isinstance(value, dict):
        print(f"{indent}dict  ({len(value)} anahtar)")
        if depth >= max_depth:
            print(f"{indent}  (max-depth={max_depth} sınırına ulaşıldı, içerik gösterilmiyor)")
            return
        items = list(value.items())
        for key, sub_value in items[:max_items]:
            print(f"{indent}  '{key}':")
            describe_value(sub_value, depth + 1, max_depth, max_items, indent + "    ")
        if len(items) > max_items:
            print(f"{indent}  ... ve {len(items) - max_items} anahtar daha (toplam {len(items)})")
        return

    if isinstance(value, (list, tuple)):
        type_name = "list" if isinstance(value, list) else "tuple"
        print(f"{indent}{type_name}  ({len(value)} eleman)")
        if depth >= max_depth:
            print(f"{indent}  (max-depth={max_depth} sınırına ulaşıldı, içerik gösterilmiyor)")
            return
        for i, sub_value in enumerate(value[:max_items]):
            print(f"{indent}  [{i}]:")
            describe_value(sub_value, depth + 1, max_depth, max_items, indent + "    ")
        if len(value) > max_items:
            print(f"{indent}  ... ve {len(value) - max_items} eleman daha (toplam {len(value)})")
        return

    repr_str = repr(value)
    if len(repr_str) > 200:
        repr_str = repr_str[:200] + "...(kısaltıldı)"
    print(f"{indent}{type(value).__name__}  değer={repr_str}")


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[probe_fedavg] Yükleniyor: {args.path}")

    with open_readonly(args.path) as f:
        raw = torch.load(f, map_location="cpu", weights_only=False)

    print(f"[probe_fedavg] Üst seviye tip: {type(raw).__name__}")
    describe_value(raw, depth=0, max_depth=args.max_depth, max_items=args.max_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
