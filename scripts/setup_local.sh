#!/usr/bin/env bash
# Yerel geliştirme ortamı kurulumu: ezkl, solc-select (0.8.20), foundry.
# Linux ve macOS için. Her adımdan sonra kurulumun gerçekten olduğu
# doğrulanır — sessizce geçilmez, hata varsa betik durur.
set -euo pipefail

fail() {
    echo "HATA: $1" >&2
    exit 1
}

check_cmd() {
    local name="$1"
    command -v "$name" >/dev/null 2>&1 || fail "'$name' PATH'te bulunamadı, kurulum başarısız görünüyor."
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

echo "== ezkl CLI =="
if ! command -v ezkl >/dev/null 2>&1; then
    echo "ezkl CLI bulunamadı, kuruluyor..."
    curl -fsSL https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash \
        || fail "ezkl CLI kurulum betiği başarısız."
    export PATH="$HOME/.ezkl/bin:$PATH"
fi
check_cmd ezkl
ezkl --version || fail "ezkl kuruldu ama --version çalışmadı."
echo "ezkl OK: $(ezkl --version)"

echo "== solc-select (0.8.20) =="
if ! command -v solc-select >/dev/null 2>&1; then
    pip install solc-select || fail "solc-select pip kurulumu başarısız."
fi
check_cmd solc-select
solc-select install 0.8.20 || fail "solc 0.8.20 kurulamadı."
solc-select use 0.8.20 || fail "solc 0.8.20 aktive edilemedi."
check_cmd solc
INSTALLED_SOLC="$(solc --version | grep -o '0\.8\.20' || true)"
[ -n "$INSTALLED_SOLC" ] || fail "solc aktif sürümü 0.8.20 değil: $(solc --version)"
echo "solc 0.8.20 OK"

echo "== Foundry (forge/anvil/cast) =="
if ! command -v foundryup >/dev/null 2>&1; then
    curl -fsSL https://foundry.paradigm.xyz | bash || fail "foundryup kurulum betiği başarısız."
    export PATH="$HOME/.foundry/bin:$PATH"
fi
foundryup || fail "foundryup çalıştırılamadı."
check_cmd forge
check_cmd anvil
check_cmd cast
forge --version || fail "forge kuruldu ama --version çalışmadı."
anvil --version || fail "anvil kuruldu ama --version çalışmadı."
cast --version || fail "cast kuruldu ama --version çalışmadı."
echo "Foundry OK"

echo "== Kurulum tamamlandı, tüm araçlar doğrulandı =="
