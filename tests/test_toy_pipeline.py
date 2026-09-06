"""Faz B kabul testi: küçük MLP -> ONNX -> ezkl -> Solidity verifier ->
anvil -> web3 doğrulama, uçtan uca.

Bu ortamda (yerel, Windows) `ezkl`/`anvil`/`solc` yoksa test SKIP olur —
bu bir hata değildir, `pytest`'in yeşil kalmasını sağlar (CLAUDE.md
madde 7/8). Colab'da (hepsi kurulu) gerçekten çalışır.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("ezkl", reason="ezkl kurulu değil — Faz B testi sadece Colab'da (ya da ezkl+solc+anvil kurulu bir ortamda) koşar.")

if shutil.which("anvil") is None:
    pytest.skip("anvil PATH'te yok — Faz B testi sadece Colab'da (scripts/setup_colab.sh sonrası) koşar.", allow_module_level=True)

if shutil.which("solc") is None:
    pytest.skip("solc PATH'te yok — Faz B testi sadece Colab'da (scripts/setup_colab.sh sonrası) koşar.", allow_module_level=True)

from chain.anvil import AnvilProcess  # noqa: E402
from circuits.toy_pipeline import run_full_pipeline  # noqa: E402


def test_toy_pipeline_end_to_end():
    with tempfile.TemporaryDirectory(prefix="zk_toy_pipeline_") as tmp_dir:
        work_dir = Path(tmp_dir)

        with AnvilProcess() as anvil:
            report = run_full_pipeline(work_dir, anvil)

        used_native = report.get("used_native_ezkl_deploy", False)

        print("\n=== Faz B raporu ===")
        for step, elapsed in report["steps"].items():
            print(f"  {step:<28} {elapsed:>8.3f}s")
        if used_native:
            print(f"  {'used_native_ezkl_deploy':<28} True (bkz. solc_fallback_reason)")
            print(f"  {'solc_fallback_reason':<28} {report.get('solc_fallback_reason')}")
        else:
            print(f"  {'used_solc_strategy':<28} {report.get('used_solc_strategy')}")
            print(f"  {'deployed_bytecode_size':<28} {report.get('deployed_bytecode_size')}")
            print(f"  {'exceeds_eip170':<28} {report.get('exceeds_eip170')}")
        print(f"  {'contract_address':<28} {report['contract_address']}")
        print(f"  {'deploy_gas':<28} {report['deploy_gas']}")
        print(f"  {'verify_gas':<28} {report['verify_gas']}")

        # exceeds_eip170/deploy_gas/verify_gas sadece manuel solc yolunda
        # (used_native değilken) kesin biliniyor; native ezkl yolunda gas
        # geriye dönük taramayla "en iyi çaba" bulunuyor, bulunamazsa None
        # kalır (uydurulmaz) — bu yüzden orada sert assert edilmiyor.
        if not used_native:
            assert report["exceeds_eip170"] is False
            assert report["deploy_gas"] > 0
            assert report["verify_gas"] > 0

        assert report["verified"] is True
