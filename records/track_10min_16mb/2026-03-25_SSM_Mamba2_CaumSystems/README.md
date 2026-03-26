# First Pure State-Space Model (SSM) Submission to Parameter Golf

**val_bpb: TBD** (pending 8xH100 run) | **~12 MB estimated** | 8xH100 SXM

> **This is the first pure SSM submission to Parameter Golf.** OpenAI explicitly requested state-space model entries (see "Requests for PRs" in README). We deliver a pure Mamba-2 SSD architecture with zero attention. A hybrid approach (Hymba) achieved 1.1828 BPB — this submission explores whether pure SSM can be competitive.

## Architecture

| Component | Setting |
|-----------|---------|
| Architecture | **Mamba-2 SSD** (pure SSM, no attention) |
| Layers | 10 |
| Model dim | 512 |
| SSM state dim | 64 |
| Conv width | 4 |
| Expand factor | 2x (d_inner=1024) |
| Head dim | 64 (16 SSM heads) |
| Skip connections | U-Net encoder/decoder with learned weights |
| Embedding | Tied, vocab=1024 |
| Softcap | tanh, cap=30.0 |
| Estimated params | ~17.1M (~6D^2 per layer vs ~12D^2 for transformer) |
| Estimated artifact | ~12MB (int8+zlib) |

### Parameter Budget per Block

| Component | Params |
|-----------|--------|
| in_proj (512 -> 2192) | 1,122,304 |
| conv1d (depthwise, k=4) | 5,760 |
| dt_bias + A_log + D | 48 |
| norm (d_inner) | 1,024 |
| out_proj (1024 -> 512) | 524,288 |
| block_scale | 512 |
| **Total per block** | **~1,654,000** |

## Why Mamba-2 for Parameter Golf?

1. **No attention = O(L) complexity.** Mamba processes sequences in linear time via selective state spaces, not quadratic attention. This allows efficient training with longer context in the 10-min budget.

2. **Parameter efficiency.** Published results show Mamba-3B matching Transformer-6B perplexity (40% cheaper). Each Mamba block at ~6D^2 params replaces both attention (~4D^2) and MLP (~8D^2) — halving per-layer cost.

3. **Mamba-2's SSD layer** reformulates the SSM as structured matrix multiplication (State Space Duality), achieving 2-8x speedup over Mamba-1 while maintaining quality. Critical for the 10-min constraint.

4. **Selective state spaces** are input-dependent (B, C, dt parameters vary per token), providing content-aware sequence modeling without O(L^2) cost.

## Key Design Decisions

### Pure SSM (no hybrid)
A hybrid SSM-attention approach (Hymba) already achieved 1.1828 BPB in this competition. We intentionally omit attention to provide the first **pure SSM** data point. This tests whether the selective scan mechanism alone can compete with transformers in the parameter-constrained regime.

### U-Net skip connections
Borrowed from the baseline transformer. The first 5 layers are "encoder" (storing skip activations), the last 5 are "decoder" (consuming them). Provides short-circuit gradients and multi-scale feature reuse.

### Muon optimizer for matrix params
The Muon optimizer (Newton-Schulz orthogonalization) works with any 2D parameter matrix. Mamba-2's in_proj/out_proj are 2D -> Muon applies directly. Scalar params (A_log, D, dt_bias, block_scale) use Adam.

### int8+zlib quantization
Same pipeline as baseline. All 2D float tensors -> per-row int8 with fp16 scales. Small/control tensors in fp16/fp32 passthrough. zlib level 9 compression.

## Setup and Run

```bash
# On RunPod 8xH100 pod (use official Parameter Golf template):
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Install mamba-ssm + download dataset
bash records/track_10min_16mb/2026-03-25_SSM_Mamba2_CaumSystems/setup.sh

# Run 3 seeds
for SEED in 1337 42 2025; do
    SEED=$SEED bash records/track_10min_16mb/2026-03-25_SSM_Mamba2_CaumSystems/run.sh
done
```

### Tuning

Try more layers if artifact fits: `NUM_LAYERS=12 SEED=1337 bash run.sh`
Try longer context: `TRAIN_SEQ_LEN=2048 SEED=1337 bash run.sh`
Disable torch.compile if issues: `USE_COMPILE=0 SEED=1337 bash run.sh`

## Results

| Seed | Steps | ms/step | val_bpb | RT bpb | Artifact |
|------|-------|---------|---------|--------|----------|
| 1337 | TBD | TBD | TBD | TBD | TBD |
| 42 | TBD | TBD | TBD | TBD | TBD |
| 2025 | TBD | TBD | TBD | TBD | TBD |

## Compliance

- [ ] 3 seeds run on 8xH100 SXM
- [ ] All seeds train in <= 600s
- [ ] All seeds artifact <= 16,000,000 bytes
- [ ] No test-time training on validation data
- [ ] No network calls during evaluation

## Prior Art in This Competition

| Entry | BPB | Type |
|-------|-----|------|
| SOTA (LeakyReLU² + TTT) | 1.1194 | Transformer |
| Hymba (hybrid SSM+attn) | 1.1828 | Hybrid |
| Baseline | 1.2244 | Transformer |
| **This submission** | **TBD** | **Pure SSM** |

## About CAUM Systems

[CAUM Systems](https://github.com/Blasmerit) builds passive behavioral monitoring for AI agent workflows. Our motor (v10.31.0) uses SBERT embeddings and LZ complexity analysis to detect behavioral regimes (loops, stagnation) in real-time — compression expertise that directly informed this SSM architecture choice.
