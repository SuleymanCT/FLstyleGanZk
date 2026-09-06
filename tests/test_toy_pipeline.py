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

        print("\n=== Faz B raporu ===")
        for step, elapsed in report["steps"].items():
            print(f"  {step:<28} {elapsed:>8.3f}s")
        print(f"  {'deployed_bytecode_size':<28} {report['deployed_bytecode_size']}")
        print(f"  {'exceeds_eip170':<28} {report['exceeds_eip170']}")
        print(f"  {'contract_address':<28} {report['contract_address']}")
        print(f"  {'deploy_gas':<28} {report['deploy_gas']}")
        print(f"  {'verify_gas':<28} {report['verify_gas']}")

        assert report["exceeds_eip170"] is False
        assert report["verified"] is True
        assert report["deploy_gas"] > 0
        assert report["verify_gas"] > 0
