#!/bin/bash
# Setup script for Mamba-2 SSM submission on RunPod
set -e

echo "=== CAUM Systems — Mamba-2 SSM Setup ==="

# Install mamba-ssm with CUDA kernels
pip install "mamba-ssm>=2.3.0" "causal-conv1d>=1.4.0" --no-build-isolation

# Download FineWeb dataset (1024 vocab)
cd /workspace/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

echo "=== Setup complete. Run with: ==="
echo "SEED=1337 bash run.sh"
