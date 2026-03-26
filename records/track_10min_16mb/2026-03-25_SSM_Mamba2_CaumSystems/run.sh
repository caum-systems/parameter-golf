#!/bin/bash
# Run Mamba-2 SSM training on 8xH100
set -e

# cd to the directory containing this script (submission dir)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Repo root for data paths
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

RUN_ID=mamba2_ssm_${SEED:-1337} \
DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024/" \
TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
NUM_LAYERS=${NUM_LAYERS:-10} \
MODEL_DIM=${MODEL_DIM:-512} \
D_STATE=${D_STATE:-64} \
D_CONV=${D_CONV:-4} \
EXPAND=${EXPAND:-2} \
HEADDIM=${HEADDIM:-64} \
TIE_EMBEDDINGS=1 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
MUON_MOMENTUM=0.95 \
MUON_MOMENTUM_WARMUP_START=0.85 \
MUON_MOMENTUM_WARMUP_STEPS=500 \
WARMDOWN_ITERS=1200 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-1024} \
USE_COMPILE=${USE_COMPILE:-1} \
SEED=${SEED:-1337} \
torchrun --standalone --nproc_per_node=8 train_gpt.py
