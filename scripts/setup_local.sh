#!/usr/bin/env bash
# Yerel ortam kurulumu — SADECE kod yazımı ve `pytest` birim testleri
# içindir. ezkl proving, anvil/forge, gerçek zincir işlemleri hiç
# yerelde çalışmaz, hepsi Colab'da (bkz. scripts/setup_colab.sh).
# Bu yüzden burada foundry/ezkl CLI kurulumu YOK — sadece pytest'in
# ihtiyaç duyduğu Python paketleri kurulur.
set -euo pipefail

fail() {
    echo "HATA: $1" >&2
    exit 1
}

echo "== Platform tespiti =="
OS="$(uname -s)"
case "$OS" in
    Linux*)  PLATFORM=linux ;;
    Darwin*) PLATFORM=macos ;;
    *) fail "Desteklenmeyen platform: $OS (sadece Linux/macOS)" ;;
esac
echo "Platform: $PLATFORM"

echo "== Python sürümü kontrolü =="
PY_VER="$(python3 --version 2>&1 | awk '{print $2}')"
case "$PY_VER" in
    3.11.*) echo "Python $PY_VER OK" ;;
    3.12.*) fail "Python 3.12 tespit edildi. StyleGAN3-r CUDA custom op'ları 3.12'de derlenmiyor. Python 3.11 kullan." ;;
    *) echo "UYARI: Python $PY_VER bekleneni (3.11.x) tutmuyor, devam ediliyor ama sorun çıkabilir." ;;
esac

echo "== pip bağımlılıkları =="
pip install -r "$(dirname "$0")/../requirements.txt" || fail "pip install başarısız."
python3 -c "import ezkl, onnx, onnxruntime, torch, numpy, yaml, web3, pytest" \
    || fail "Bir veya daha fazla Python paketi import edilemedi."
echo "Python bağımlılıkları OK"

echo "== Kurulum tamamlandı =="
echo "NOT: anvil/forge/solc bilerek kurulmadı — ezkl proving ve zincir"
echo "     işlemleri gerektiren testler (ör. tests/test_toy_pipeline.py)"
echo "     bu ortamda otomatik olarak 'skip' edilir. Gerçek koşum için"
echo "     Colab'da notebooks/colab_runner.ipynb kullan."
