#!/bin/bash
# =============================================================================
# Parameter Golf — H100 Deployment Script
# Config: 6×2 @ 896d (Champion Config, local BPB: 1.4808)
# Target: 8×H100 SXM, RunPod, 10 min training
# =============================================================================

set -euo pipefail

echo "============================================"
echo "  Parameter Golf — 6×2 @ 896d CHAMPION RUN"
echo "============================================"

# --- Champion Config (all defaults in code, just override what's needed) ---
export NUM_UNIQUE_BLOCKS=6
export REPEATS=2
export MODEL_DIM=896
export NUM_HEADS=8
export NUM_KV_HEADS=4
export LORA_RANK=8
export VALUE_RESIDUAL=1
export DEEP_SUP_ENABLED=1
export QAT_ENABLED=1
export PROG_SEQ_ENABLED=0
export SEED=${SEED:-1337}

# --- Training params (match #1 entry) ---
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=20000
export WARMDOWN_ITERS=3500
export WARMUP_STEPS=20
export GRAD_CLIP_NORM=0.3

# --- Muon optimizer (already in code) ---
export MUON_MOMENTUM=0.99
export MUON_BACKEND_STEPS=5
export MUON_WD=0.04
export ADAM_WD=0.04

# --- EMA + SWA (match #1) ---
export SWA_ENABLED=1
export SWA_EVERY=50

# --- TTT (test-time training) ---
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_MOMENTUM=0.9
export TTT_GRAD_CLIP=1.0

echo ""
echo "  Config: NUM_UNIQUE_BLOCKS=$NUM_UNIQUE_BLOCKS, REPEATS=$REPEATS"
echo "  Model:  ${MODEL_DIM}d, ${NUM_HEADS} heads, LoRA-${LORA_RANK}"
echo "  Batch:  ${TRAIN_BATCH_TOKENS} tokens/step, seq=${TRAIN_SEQ_LEN}"
echo "  Seed:   $SEED"
echo "  TTT:    ENABLED (lr=$TTT_LR, epochs=$TTT_EPOCHS)"
echo ""

# --- Run training (DDP across all available GPUs) ---
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "  GPUs detected: $NUM_GPUS"
echo ""

if [ "$NUM_GPUS" -gt 1 ]; then
    echo ">>> Starting distributed training on $NUM_GPUS GPUs..."
    torchrun --standalone --nproc_per_node=$NUM_GPUS \
        experiments/train_gpt_depthrecur.py
else
    echo ">>> Starting single-GPU training..."
    python experiments/train_gpt_depthrecur.py
fi

echo ""
echo "============================================"
echo "  TRAINING COMPLETE"
echo "============================================"
echo ""
echo "  Check logs/ for results and artifacts."
echo "  Artifacts should be in logs/<run_id>/"
echo ""
