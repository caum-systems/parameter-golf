#!/bin/bash
# =============================================================================
# RunPod Setup Script — Run this FIRST after deploying the pod
# =============================================================================

set -euo pipefail

echo "============================================"
echo "  RunPod Environment Setup"
echo "============================================"

# --- Clone repo ---
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/LVNETvenezuela/parameter-golf.git
fi
cd parameter-golf

# --- Install dependencies ---
pip install -q sentencepiece zstandard

# --- Download data (if not cached) ---
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo ">>> Downloading FineWeb-10B dataset..."
    python -c "
import os, subprocess
os.makedirs('data/datasets', exist_ok=True)
os.makedirs('data/tokenizers', exist_ok=True)
# The training script should handle data download automatically
# If not, uncomment and adapt:
# subprocess.run(['python', 'data/download.py'], check=True)
print('Data directory ready. Training script will download if needed.')
"
fi

# --- Verify GPU setup ---
echo ""
echo ">>> GPU Check:"
nvidia-smi -L
echo ""
echo ">>> PyTorch CUDA:"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
echo ""

# --- Check FlashAttention 3 ---
python -c "
try:
    from flash_attn_interface import flash_attn_func
    print('  FlashAttention 3: AVAILABLE')
except ImportError:
    print('  FlashAttention 3: NOT FOUND (will use FA2 or math)')
    print('  Install: pip install flash-attn --no-build-isolation')
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "  Run: bash run_h100.sh"
echo "============================================"
