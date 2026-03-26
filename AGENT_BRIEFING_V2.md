# Parameter Golf — Complete Technical Briefing V2
## For agent doing H100 testing. You know NOTHING about this project. Read everything.

---

## 1. WHAT IS PARAMETER GOLF?

OpenAI competition. **Deadline: April 30, 2026.**
- Train a language model **from scratch** in exactly **10 minutes on 8×H100 SXM GPUs**
- The saved artifact (model weights file + train_gpt.py code) must be **≤ 16 MB total**
- Score = **BPB (Bits Per Byte)** on the FineWeb validation set. **Lower BPB = better.**
- Dataset: FineWeb 10B tokens, SentencePiece BPE tokenizer with **1024 vocab size**
- Training command: `torchrun --nproc_per_node=8 train_gpt.py`
- All code is a single `train_gpt.py` file. No external model downloads. Everything trains from random init.
- Official repo: https://github.com/openai/parameter-golf
- License: MIT. Building on other submissions is the norm — every top entry does it.

## 2. CURRENT LEADERBOARD

| Rank | BPB | Key technique |
|------|-----|--------------|
| **#1** | **1.1194** | 11L/512d + LeakyReLU² + TTT + Parallel Muon + int6+LZMA |
| #2 | 1.1233 | 11L/512d + SmearGate + BigramHash + QAT + GPTQ-lite |
| Baseline | 1.2244 | Standard 11L/512d transformer |

The gap between #1 and #2 is only **0.0039 BPB** — extremely tight.

---

## 3. THE #1 SUBMISSION (PR #549) — OUR BASE

We forked the #1 submission. Here is EXACTLY what it does, component by component:

### 3.1 Architecture (GPT class, 27M params, 11L/512d)

**Transformer basics:**
- 11 transformer layers, model_dim=512, 8 attention heads (GQA: 4 KV heads), head_dim=64
- MLP: 3× expansion (hidden_dim=1536) with **LeakyReLU(negative_slope=0.5) squared** activation
  - `x = F.leaky_relu(F.linear(x, up_w), negative_slope=0.5); return F.linear(x.square(), down_w)`
- Tied embeddings: `tok_emb.weight` is reused as the output projection (saves an entire vocab×dim matrix)
- Logit softcap at 30.0: `logits = 30 * tanh(raw_logits / 30)` prevents logit explosion

**Positional encoding:**
- Partial RoPE: only first 16 of 64 head dimensions get rotary encoding. Rest are position-free.
- Standard RoPE base=10000

**Per-layer learned scalars (critical for training stability):**
- `attn_scale`: per-dim scalar multiplied onto attention output
- `mlp_scale`: per-dim scalar multiplied onto MLP output
- `resid_mix`: 2×dim parameter. Mixes current hidden state `x` with initial embedding `x0`:
  - `x_in = mix[0] * x + mix[1] * x0` (lets deep layers directly access raw token info)
- `ln_scale`: `1/sqrt(layer_idx+1)` multiplied onto normalized input before attention/MLP
  - Layer 0 gets factor 1.0, layer 10 gets factor 0.30. Stabilizes deep layers.
- `q_gain`: per-head scalar on queries after QK-RMSNorm (init 1.5)

**U-Net skip connections:**
- 5 encoder layers (0-4) push to a stack, 6 decoder layers (5-10) pop from the stack
- `skip_weights`: learned per-dim scaling on the skip connection
- `x = x + skip_weights[i] * skips.pop()`

**Special modules:**
- **SmearGate**: `g = sigmoid(gate); x = (1-g)*x + g*x_prev` — mixes current token with previous token. Applied ONCE after embedding, before the transformer layers. `gate` is a learned per-dim parameter (dim-sized vector, init 0 → sigmoid=0.5).
- **BigramHashEmbedding**: XOR hash of adjacent token IDs → lookup in 2048-entry embedding table → project to model_dim → scale by learned scalar (init 0.05). Gives the model bigram statistics for free.
- **XSA (eXclude Self Attention)**: On last 4 layers only. After attention, subtracts the self-value projection: `y = y - proj(y, normalize(v))`. Forces attention to look at OTHER tokens, not itself.
- **ValueEmbedding**: On layers 9,10 only. Reinjects token identity into the value stream: `v = F.linear(x, v_w) + ve_shared(token_ids) * scale`. Shared embedding table, per-layer learned scale.
- **QK RMSNorm**: Query and Key are RMS-normalized per-head BEFORE RoPE application.

### 3.2 Weight Storage — Banks

All Q,K,V,O,MLP weights are stored as 3D "bank" tensors (not per-layer nn.Linear):
```
qo_bank:       [2×num_layers, model_dim, model_dim]     # first half Q, second half O
kv_bank:        [2×num_layers, kv_dim, model_dim]         # first half K, second half V
mlp_up_bank:    [num_layers, mlp_dim, model_dim]
mlp_down_bank:  [num_layers, model_dim, mlp_dim]
```
This format enables batched Newton-Schulz orthogonalization in the Muon optimizer (operates on the entire bank tensor at once).

### 3.3 Optimizer — Parallel Muon (THIS IS CRITICAL)

Two optimizer groups with completely different update rules:

**Group 1: Muon (for the 4 bank tensors)**
- Newton-Schulz 5 orthogonalization: projects gradient onto the nearest orthogonal matrix
- Coefficients: a=3.4445, b=-4.7750, c=2.0315 (5 iterations)
- Momentum 0.99 (warmup from 0.92 over 1500 steps)
- LR = 0.025
- Weight decay = 0.04
- **No DDP!** Manual 3-phase overlapped communication:
  1. After backward: launch async reduce-scatter for all banks (biggest first)
  2. While reduce-scatter is in-flight: run Adam steps on small params
  3. Wait for reduce-scatter, run local NS5 on the shard, launch async all-gather
  4. Each all-gather overlaps with next bank's NS5 computation

**Group 2: AdamW (for everything else)**
- Token embeddings: lr=0.035
- Scalar/control params: lr=0.025
- Betas: (0.9, 0.95), eps=1e-8
- Weight decay: 0.04
- Non-bank gradients are manually all-reduced (dist.all_reduce AVG) before stepping

### 3.4 Training Loop

1. **Kernel warmup**: 20 steps of training, then FULL RESET of model weights + optimizer states. This lets torch.compile and CUDA warm up without affecting the real training.
2. **Main training**: ~14,000 steps in 10 minutes on 8×H100
   - Batch: 786,432 tokens per step, seq_len=2048
   - Gradient accumulation: 8 // world_size = 1 micro-step per GPU
   - Gradient clipping: max_norm=0.3
3. **Wallclock-aware LR warmdown**: LR linearly decays to 0, timed to reach 0 at exactly the 10-minute mark. Uses elapsed wall-clock time, not step count.
4. **Late QAT**: When LR scale < 0.15, enables int6 STE (Straight-Through Estimator) simulation in all CastedLinear layers. This pre-adapts the model to quantization noise before saving.
5. **EMA**: Exponential moving average with decay=0.997, applied to final weights.
6. **SWA**: Stochastic Weight Averaging — snapshots every 50 steps when LR scale < 0.2. Averaged at end.

### 3.5 Quantization Pipeline (for 16MB artifact)

After training finishes:
1. `_unbank_state_dict()`: Convert 3D bank tensors → individual 2D per-layer tensors
2. `mixed_quantize_int6()`:
   - Attention + MLP weights → **int6** per-row quantization (clip range [-31,31], searches 5 percentiles for lowest MSE)
   - Embeddings → **int8** per-row quantization
   - Small control tensors (<65536 elements) → **float16** passthrough
3. `lzma.compress(preset=6)`: LZMA compression on the serialized torch.save blob
4. Verification: load the compressed artifact, dequantize, rebuild model, evaluate — this roundtrip BPB is what gets scored

### 3.6 Evaluation

**Standard eval**: Full-sequence cross-entropy loss, converted to BPB using SentencePiece byte counts.

**Sliding window eval** (used for final score): stride=64, seq_len=2048. Each token scored with maximum available context.

**TTT (Test-Time Training)** — legal "score-first" protocol:
- Divide validation data into 32K-token chunks
- For each chunk: SCORE it first (inference_mode, accumulate BPB), THEN train on it
- SGD with momentum 0.9, lr=0.002, 3 epochs per chunk
- Freezes first 2 transformer blocks during TTT
- Every token's BPB is locked in BEFORE any parameter update that could use that token = legal

---

## 4. OUR INNOVATION: DEPTH RECURRENCE + LoRA

### 4.1 The Core Idea

Instead of 11 unique layers with 11 unique weight banks, we use **4 shared weight banks repeated 3 times = 12 effective layers**. This slashes the parameter count for the heavy bank tensors by ~3×, freeing massive space in the 16MB artifact to go wider.

```
Original #1:  11 unique banks × 512d = 27M params, ~15MB artifact
Our model:     4 shared banks × 768d = 22.4M params, ~10.7MB artifact (5.3MB headroom!)
```

### 4.2 How It Works

**Bank indexing**: Layer `i` uses bank `i % num_unique_blocks`. So layers 0,4,8 share bank 0; layers 1,5,9 share bank 1; etc.

**Per-position LoRA**: Each of the 12 effective layers gets its own low-rank adapters to specialize. Without LoRA, layers 0/4/8 would be identical — LoRA gives each position a unique "personality" while sharing the heavy base weights.

**What has LoRA (rank 8, per all 12 positions):**
- Q projection: `q_w = bank_q[bi] + lora_q_b[i] @ lora_q_a[i]` — [768,8] @ [8,768]
- K projection: `k_w = bank_k[bi] + lora_kv_b[i] @ lora_kv_a[i]`
- V projection: `v_w = bank_v[bi] + lora_kv_b[12+i] @ lora_kv_a[12+i]`
- MLP_up: `up_w = bank_up[bi] + lora_up_b[i] @ lora_up_a[i]` — [2304,8] @ [8,768]

**What does NOT have LoRA:**
- O projection (output), MLP_down — these share the exact bank weights across positions

**LoRA initialization**: A matrices get kaiming_uniform, B matrices get zeros → LoRA starts as zero delta (model behaves as pure shared banks at init)

**LoRA total params: 639K** (tiny compared to 20.4M in banks)

### 4.3 What Each Position Owns Uniquely

Each of the 12 Block modules has its own (NOT shared):
- `attn_scale` (per-dim, 768 params)
- `mlp_scale` (per-dim, 768 params)
- `resid_mix` (2×768 params)
- `q_gain` (per-head, 12 params)
- `ln_scale_factor` (computed from effective layer index 0-11, not bank index)
- `attn_norm`, `mlp_norm` (RMSNorm, no learned params)

### 4.4 Bank Tensor Shapes

```python
qo_bank:       [2×4, 768, 768]     = 4,718,592 params
kv_bank:        [2×4, 256, 768]     = 1,572,864 params
mlp_up_bank:    [4, 2304, 768]      = 7,077,888 params
mlp_down_bank:  [4, 768, 2304]      = 7,077,888 params
# Total banks: 20,447,232 params (but only 4 unique sets!)
```

### 4.5 Model Dimensions

```
model_dim = 768
num_heads = 12 (query heads)
num_kv_heads = 4 (GQA: 3 query heads per KV head)
head_dim = 768 / 12 = 64
kv_dim = 4 × 64 = 256
mlp_dim = 3 × 768 = 2304
```

---

## 5. BUGS WE FOUND AND FIXED

### 5.1 CRITICAL: Muon Gradient Scaling (was BROKEN, now FIXED)

**The problem**: With depth recurrence, bank[0] receives backward gradients from layers 0, 4, AND 8. PyTorch accumulates these — so `bank.grad` is 3× larger than what Muon expects for a single-use weight. Newton-Schulz orthogonalization was designed for single-layer gradients. Feeding it 3× amplified gradients destabilizes the optimizer.

**The fix** (in training loop, after grad_clip_norm, before Muon step):
```python
if args.num_unique_blocks > 0 and args.repeats > 1:
    rep_scale = 1.0 / args.repeats  # = 1/3
    for p in matrix_params:  # the 4 bank tensors
        if p.grad is not None:
            p.grad.mul_(rep_scale)
```
This is applied AFTER gradient clipping but BEFORE `optimizer_muon.launch_reduce_scatters()`.

**Impact**: Without this fix, training would likely diverge or produce suboptimal convergence on H100 at 14K steps. This is the single most important bug fix.

### 5.2 VERIFIED OK: ln_scale uses effective index

`Block` is created with `layer_idx=i` where `i in range(num_layers)` = 0 to 11. So `ln_scale_factor = 1/sqrt(i+1)` correctly uses the effective depth, NOT the bank index. Layer 0 gets 1.0, layer 11 gets 0.289. This was already correct.

### 5.3 VERIFIED OK: SmearGate applied once, not recursively

`self.smear(x)` is called once after embedding, before the transformer loop. It does NOT run inside each layer pass. This is correct — recursive SmearGate would wash out token identity.

---

## 6. IMPROVEMENTS WE IMPLEMENTED (in `train_gpt_depthrecur.py`)

### 6.1 Deep Supervision (DEEP_SUP_ENABLED=1, DEEP_SUP_WEIGHT=0.1)

At every repetition boundary (after layers 4 and 8 in a 4×3 config), compute an auxiliary cross-entropy loss using the tied embeddings as a free output projection:

```python
# At rep boundary (layer 4, layer 8):
x_aux = F.rms_norm(x, (x.size(-1),))
aux_logits = F.linear(x_aux, self.tok_emb.weight)  # 0 extra params!
aux_logits = softcap * tanh(aux_logits / softcap)
aux_loss = cross_entropy(aux_logits, targets)
```

Weights decay exponentially: `weight = 0.1 * 0.5^(num_aux - idx)`. For 4×3 this gives:
- After layer 4: weight = 0.05
- After layer 8: weight = 0.1

**Rationale**: Forces shared banks to receive direct gradient signal at every repetition. Without this, only the final layer's loss gradient propagates back through 12 layers of the same 4 banks — vanishing gradient problem. Deep supervision gives 3× the direct gradient signal.

**Cost**: Zero extra parameters. Small compute cost (one extra matmul per boundary per step). Only active during training.

### 6.2 LoRA on K and V (NEW — was only on Q and MLP_up before)

Added LoRA adapters to Key and Value projections:
```python
lora_kv_a: [2×12, 8, 768]   # first 12 = K adapters, second 12 = V adapters
lora_kv_b: [2×12, 256, 8]
# Extra params: 196,608 (tiny)
```

**Rationale**: K controls what information is "advertised" to queries. V controls what payload is extracted. With shared banks, all positions had identical K/V behavior — only Q and MLP varied. Now every position can specialize what it attends to AND what it returns.

### 6.3 LoRA-Only TTT (bank weights FROZEN during test-time training)

Changed the TTT strategy from "train everything except first 2 blocks" to "freeze ALL 4 bank tensors, only train LoRA + scalars + embeddings":

```python
# Frozen during TTT: qo_bank, kv_bank, mlp_up_bank, mlp_down_bank (~20M params)
# Trainable during TTT: lora_q, lora_kv, lora_up, scalars, embeddings (~2M params)
```

Also changed optimizer from SGD to AdamW (converges faster on small param sets).

**Rationale**: TTT with 22M params is slow and risks catastrophic forgetting. LoRA-only TTT is 10× faster (backprop through ~2M params vs ~20M), can afford 5-8 epochs per chunk vs 3, and preserves the core grammar in the frozen banks.

### 6.4 LZMA Neuron Permutation

Before quantizing MLP weights, sort the hidden neurons by L2 norm:
```python
def _permute_mlp_neurons(up_w, down_w):
    norms = up_w.float().norm(dim=1)  # norm per hidden neuron
    perm = norms.argsort()
    return up_w[perm], down_w[:, perm]
```

MLP is permutation-invariant (swapping row i of up_w and column i of down_w is mathematically identical). Sorting creates spatially smooth weight matrices that LZMA compresses better. Expected savings: 5-15% on trained weights (negligible on random weights, which we verified).

---

## 7. ALL EXPERIMENTAL DATA

### 7.1 Ablation Sweep (500 steps, local RTX 4070 SUPER, 11L/512d)

Each config tests ONE SOTA technique added to the baseline:

| Config | Params | Loss (last 5 avg) | Min loss | Delta vs baseline | ms/step | Artifact |
|--------|--------|-------------------|----------|-------------------|---------|----------|
| **FULL_SOTA** | 27.0M | **4.0427** | 3.837 | **-0.1197** | 250 | 14.5MB |
| +bigram_hash | 26.8M | 4.1124 | 3.901 | -0.0499 | 80 | 14.7MB |
| +leaky_relu_sq | 26.5M | 4.1434 | 3.997 | -0.0190 | 80 | 14.2MB |
| baseline_11L_3x | 26.5M | 4.1623 | 4.019 | — | 108 | 13.8MB |
| +ln_scale | 26.5M | 4.1660 | 3.987 | -0.0037 | 77 | 13.8MB |
| +rope16 | 26.5M | 4.1623 | 4.028 | +0.0000 | 81 | 13.6MB |
| +ve_9_10 | 26.7M | 4.1585 | 4.021 | -0.0038 | 219 | 14.1MB |
| +xsa4 | 26.5M | 4.1718 | 4.018 | +0.0095 | 80 | 13.9MB |
| +smeargate | 26.5M | 4.2114 | 4.058 | **+0.0492** | 78 | 13.5MB |

**Key findings**:
- BigramHash is the single most impactful technique (-0.0499)
- LeakyReLU² is the second most impactful (-0.0190)
- SmearGate HURTS at 500 steps (+0.0492) — it likely needs 2000+ steps to help
- FULL_SOTA synergy is real: combined techniques (-0.1197) > sum of individual gains
- VE adds significant latency (219ms vs 108ms) for minimal gain

### 7.2 Depth Recurrence Sweep (500 steps, all with SOTA techniques)

| Config | Unique×Reps | Dim | Params | Loss | Delta vs baseline | Artifact | ms/step |
|--------|-------------|-----|--------|------|-------------------|----------|---------|
| **rec_4x3_512_sota** | 4×3 | 512 | 10.3M | **4.0944** | **-0.0984** | 5.5MB | 153 |
| rec_6x2_640_sota | 6×2 | 640 | 23.1M | 4.1065 | -0.0566 | 10.3MB | 210 |
| rec_5x3_640_sota | 5×3 | 640 | 19.5M | 4.1071 | -0.0559 | 8.5MB | 222 |
| rec_4x3_640_sota | 4×3 | 640 | 15.8M | 4.1227 | -0.0396 | 7.5MB | 179 |
| baseline_11L_3x | 11×1 | 512 | 26.5M | 4.1930 | — | 13.1MB | — |
| rec_4x3_704_sota | 4×3 | 704 | — | ERROR | — | — | — |

**The 704d error**: `shape '[8, 256, 11, 64]' is invalid` — 704/11 heads = non-integer head_dim. Invalid config.

**Key findings**:
- **4×3@512 wins at 500 steps** (lowest loss AND smallest artifact)
- BUT: wider models (640d, 768d) are still converging at 500 steps — they likely win at 14K+ steps on H100
- 6×2 (more unique blocks, fewer reps) vs 4×3 (fewer blocks, more reps): similar loss but 6×2 has 2× the artifact size
- Depth recurrence WORKS: even 10.3M params beats the 26.5M baseline

### 7.3 Our Current Model (post-improvements, verified)

```
Config:          4×3@768d, LoRA rank 8 on Q/K/V/MLP_up
Total params:    22,439,316
Bank params:     20,447,232 (4 unique sets of banks)
LoRA params:       638,976 (12 positions × 4 projections × rank 8)
Other params:    1,353,108 (embeddings, scalars, BigramHash, VE, etc.)
Artifact est:    ~10.7MB (5.3MB headroom under 16MB limit)
Training test:   loss 7.47 → 6.67 in 20 steps (73ms/step on RTX 4070 SUPER)
```

---

## 8. WHAT WE HAVE NOT TESTED YET

These are improvements identified but not yet implemented. They need H100 validation:

### 8.1 Progressive Sequence Length

Start training with seq_len=512 (4× more sequences per batch, faster steps), ramp to 1024 mid-training, then 2048 for final phase. Expected: +15-20% more total steps in 10 minutes.

**DANGER**: Changing seq_len triggers `torch.compile` cache miss (~45 seconds penalty per change). Possible workaround: keep tensor shape at 2048, pack multiple short sequences with block-diagonal attention mask.

### 8.2 TTT Hyperparameter Tuning

Current: lr=0.002, 3 epochs, 32K chunks, SGD+momentum
Proposed: lr=0.003, 5 epochs, 16K chunks, AdamW (already changed to AdamW in code)

With LoRA-only TTT, we can afford many more epochs because backprop is 10× faster.

### 8.3 In-Artifact Checkpoint Ensemble

We have 5.3MB of headroom. Could save TWO checkpoints (e.g., step 12000 and final), LZMA will deduplicate shared weights. At eval time, average their logits. Mathematically guaranteed BPB drop.

### 8.4 Wider Model (896d or 1024d)

With only 10.7MB artifact, we could try 896d (16 heads, 4 KV heads → head_dim=56, BUT 56 must be even for RoPE). Valid options:
- 768d, 12 heads → head_dim=64 ✓ (current)
- 896d, 14 heads → head_dim=64 ✓ (if 14 % num_kv_heads == 0 → need kv_heads=2 or 7 or 14)
- 1024d, 16 heads → head_dim=64 ✓ (kv_heads=4 works: 16/4=4 ✓)

**1024d would increase bank params to 36.4M** — but they're shared (4 unique sets), so artifact is still ~14.5MB. Tight but fits.

### 8.5 Palindromic Routing (U-Net-aware recurrence)

Instead of `1-2-3-4 → 1-2-3-4 → 1-2-3-4`, route as `1-2-3-4 → 4-3-2-1` (or `1-2-3-4-5-6 → 6-5-4-3-2-1`). This mirrors the U-Net structure: early blocks process raw tokens AND project to logits, deep blocks handle semantics.

Would require changing `_get_bank_weights` to use a lookup table instead of `i % num_unique_blocks`.

---

## 9. KNOWN RISKS ON H100

1. **Muon convergence**: Even with the gradient scaling fix (÷3), the shared bank optimization dynamics are untested at 14K steps. Watch for loss spikes after step 5000.

2. **LoRA vs bank learning rate balance**: Banks use Muon (lr=0.025), LoRA uses Adam (lr=0.025). The LoRA might need a higher LR (2-3×) to keep up with Muon's aggressive orthogonal updates.

3. **Deep supervision + torch.compile**: The auxiliary loss computation at rep boundaries adds dynamic control flow. If `torch.compile(fullgraph=True)` breaks, set `DEEP_SUP_ENABLED=0` as fallback.

4. **Memory**: 768d model with 12 layers, batch 786K tokens, seq_len 2048 = ~48 sequences per GPU. Fits in H100 80GB but may be tight with activation checkpointing disabled.

5. **KV LoRA interaction with XSA**: XSA subtracts the self-value projection. If V differs per position (via LoRA), the XSA subtraction may behave differently across positions. Could help or hurt — unknown.

---

## 10. FILE LOCATIONS

```
parameter-golf/
├── experiments/
│   ├── train_gpt_depthrecur.py          ← OUR FORK (the main file to test)
│   ├── sweep_depth_recurrence.py        ← sweep runner script
│   ├── ablation_results.json            ← ablation data
│   └── depth_recurrence_results.json    ← recurrence sweep data
├── records/track_10min_16mb/
│   └── 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/
│       └── train_gpt.py                 ← ORIGINAL #1 (for reference)
├── data/
│   ├── datasets/fineweb10B_sp1024/      ← training + val shards
│   └── tokenizers/fineweb_1024_bpe.model
├── AGENT_BRIEFING.md                    ← previous version of this doc
├── AGENT_BRIEFING_V2.md                 ← THIS FILE
└── APPLICATION_READY.md                 ← RunPod credit form text
```

---

## 11. HOW TO RUN LOCALLY (for quick tests)

```bash
cd parameter-golf

# Quick 100-step test with our config:
NUM_UNIQUE_BLOCKS=4 REPEATS=3 LORA_RANK=8 \
NUM_LAYERS=12 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 \
BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_LAYERS=10,11 DEEP_SUP_ENABLED=1 \
ITERATIONS=100 TRAIN_LOG_EVERY=10 \
python experiments/train_gpt_depthrecur.py

# On 8×H100:
torchrun --nproc_per_node=8 experiments/train_gpt_depthrecur.py
```

All hyperparameters are controlled via environment variables. See the `Hyperparameters` class at the top of `train_gpt_depthrecur.py`.

---

## 12. WHAT WE NEED FROM YOU

Run experiments on H100. Specific priority order:

1. **Run A**: Our fork as-is (4×3@768d, all defaults). Get the baseline BPB number.
2. **Run B**: Same but with `DEEP_SUP_ENABLED=0` — measure if deep supervision helps or hurts at scale.
3. **Run C**: If A > 1.15 BPB, try `MODEL_DIM=1024 NUM_HEADS=16` — wider model with more headroom.
4. **Run D**: Best of above + `TTT_ENABLED=1 TTT_LR=0.003 TTT_EPOCHS=5 TTT_CHUNK_TOKENS=16384` — add test-time training.

Report: final BPB, training loss curve, artifact size, any crashes or warnings.

---

## 13. ENV VAR QUICK REFERENCE

| Variable | Default | What it does |
|----------|---------|-------------|
| NUM_UNIQUE_BLOCKS | 4 | Number of shared weight banks |
| REPEATS | 3 | How many times banks are reused |
| LORA_RANK | 8 | LoRA adapter rank (0 = disabled) |
| MODEL_DIM | 768 | Model width |
| NUM_HEADS | 12 | Query attention heads |
| NUM_KV_HEADS | 4 | Key/Value attention heads (GQA) |
| NUM_LAYERS | 12 | Effective layers (should = unique × repeats) |
| DEEP_SUP_ENABLED | 1 | Auxiliary losses at rep boundaries |
| DEEP_SUP_WEIGHT | 0.1 | Weight for deepest aux loss (halved per earlier boundary) |
| TTT_ENABLED | 0 | Test-time training during eval |
| TTT_LR | 0.002 | TTT learning rate |
| TTT_EPOCHS | 3 | TTT epochs per chunk |
| TTT_CHUNK_TOKENS | 32768 | TTT chunk size |
| BIGRAM_VOCAB_SIZE | 2048 | BigramHash embedding table size (0 = disabled) |
| XSA_LAST_N | 4 | XSA on last N layers (0 = disabled) |
| VE_ENABLED | 1 | ValueEmbedding |
| VE_LAYERS | 10,11 | Which layers get ValueEmbedding |
| MAX_WALLCLOCK_SECONDS | 600 | Training time limit (seconds) |
| SEED | 1337 | Random seed |
| LATE_QAT_THRESHOLD | 0.15 | Enable int6 QAT when LR scale drops below this |
| SWA_ENABLED | 1 | Stochastic Weight Averaging |
| TRAIN_SEQ_LEN | 2048 | Training sequence length |
| TRAIN_BATCH_TOKENS | 786432 | Global batch size in tokens |
