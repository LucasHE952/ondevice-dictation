#!/usr/bin/env bash
# setup.sh — One-command setup for ondevice-dictation
#
# What this script does:
#   1. Checks hardware (Apple Silicon required)
#   2. Checks macOS version (13+ required)
#   3. Checks Python version (3.11+ required)
#   4. Creates and activates a virtual environment
#   5. Installs Python dependencies
#   6. Downloads Voxtral Realtime weights from Hugging Face (~4GB)
#   7. Verifies the model loaded correctly
#   8. Prints permission setup instructions

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Colour

ok()   { echo -e "${GREEN}✓${NC}  $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✗${NC}  $*" >&2; }
info() { echo -e "${BLUE}→${NC}  $*"; }

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║       ondevice-dictation — Setup Script       ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ── 1. Hardware check ───────────────────────────────────────────────────────
info "Checking hardware…"

ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    err "Apple Silicon (arm64) required. This machine reports: $ARCH"
    err "Intel Macs are not supported — MLX only runs on Apple Silicon."
    exit 1
fi
ok "Apple Silicon detected ($ARCH)"

# Check unified memory (8GB minimum)
MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
if [[ $MEM_GB -lt 8 ]]; then
    err "Minimum 8GB unified memory required. Detected: ${MEM_GB}GB"
    exit 1
elif [[ $MEM_GB -lt 16 ]]; then
    warn "16GB recommended for best performance. Detected: ${MEM_GB}GB"
else
    ok "Unified memory: ${MEM_GB}GB"
fi

# ── 2. macOS version check ──────────────────────────────────────────────────
info "Checking macOS version…"

MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
if [[ $MACOS_MAJOR -lt 13 ]]; then
    err "macOS 13 (Ventura) or later required. Detected: $MACOS_VERSION"
    exit 1
fi
ok "macOS $MACOS_VERSION"

# ── 3. Python version check ─────────────────────────────────────────────────
info "Checking Python version…"

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        VERSION=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [[ $MAJOR -ge 3 && $MINOR -ge 11 ]]; then
            PYTHON_CMD="$cmd"
            ok "Python $VERSION found at $(command -v $cmd)"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    err "Python 3.11 or later is required but was not found."
    info "Install via Homebrew: brew install python@3.12"
    exit 1
fi

# ── 4. Virtual environment ──────────────────────────────────────────────────
info "Setting up virtual environment…"

VENV_DIR="$(pwd)/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    ok "Created virtual environment at .venv/"
else
    ok "Virtual environment already exists at .venv/"
fi

# Activate for the remainder of this script
source "$VENV_DIR/bin/activate"

# Upgrade pip silently
pip install --upgrade pip --quiet

# ── 5. Install dependencies ─────────────────────────────────────────────────
info "Installing Python dependencies (this may take a few minutes)…"
pip install -r requirements.txt
ok "Dependencies installed"

# ── 6. Download Voxtral weights ─────────────────────────────────────────────
info "Downloading Voxtral Realtime weights from Hugging Face (~4GB)…"
info "This is a one-time download. The app runs fully offline after this."
echo ""

MODEL_DIR="$HOME/.cache/ondevice-dictation/models/voxtral-realtime"
REPO_ID="mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"

python3 - <<EOF
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path("$MODEL_DIR")
repo_id = "$REPO_ID"

if model_dir.exists() and any(model_dir.iterdir()):
    print(f"  Model weights already present at {model_dir}")
    print("  Skipping download.")
    sys.exit(0)

model_dir.mkdir(parents=True, exist_ok=True)
print(f"  Downloading {repo_id} to {model_dir}…")
snapshot_download(
    repo_id=repo_id,
    local_dir=str(model_dir),
    ignore_patterns=["*.bin", "*.pt"],  # MLX weights only; skip PyTorch files
)
print("  Download complete.")
EOF

ok "Voxtral weights ready at $MODEL_DIR"

# ── 7. Quick model verification ─────────────────────────────────────────────
info "Verifying model can be loaded (compiles MLX kernels on first run — ~30s)…"

python3 - <<'EOF'
import sys
sys.path.insert(0, "src")
from pathlib import Path
from config.defaults import MODEL_LOCAL_DIR

try:
    from mlx_lm import load
    model, processor = load(str(MODEL_LOCAL_DIR))
    print("  Model loaded successfully.")
except Exception as exc:
    print(f"  ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
EOF

ok "Model verification passed"

# ── 8. Permission instructions ──────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║         Required macOS Permissions            ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo "The app needs two macOS permissions to function:"
echo ""
echo "  1. MICROPHONE ACCESS"
echo "     → macOS will prompt you automatically on first run."
echo ""
echo "  2. ACCESSIBILITY ACCESS  ⚠  (must be granted manually)"
echo "     → macOS will never prompt for this automatically."
echo "     → Steps:"
echo "        a. Open System Settings → Privacy & Security → Accessibility"
echo "        b. Click the + button"
echo "        c. Add your Terminal app (or the Python binary)"
echo "        d. Enable the toggle"
echo ""
echo "     Without Accessibility access, text injection will not work."
echo ""

# ── Done ────────────────────────────────────────────────────────────────────
echo "╔═══════════════════════════════════════════════╗"
echo "║                 Setup Complete!               ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo "To run the Phase 1 smoke test:"
echo ""
echo "    source .venv/bin/activate"
echo "    python src/main.py --phase1"
echo ""
echo "Speak for 5 seconds and see the transcription printed to the terminal."
echo ""
