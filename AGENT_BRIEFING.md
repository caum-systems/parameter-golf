# Parameter Golf — Full Agent Briefing
## For brainstorming improvements. Think outside the box.

---

## WHAT IS PARAMETER GOLF?
OpenAI competition. **Deadline: April 30, 2026.**
- Train a language model from scratch in **10 minutes on 8×H100 SXM GPUs**
- Artifact (model weights + code) must be **≤ 16 MB**
- Score = **BPB (Bits Per Byte)** on FineWeb validation set. **Lower = better.**
- Dataset: FineWeb 10B tokens, SentencePiece BPE tokenizer with **1024 vocab**
- Training: `torchrun --nproc_per_node=8 train_gpt.py`
- Rules: https://github.com/openai/parameter-golf

## LEADERBOARD (current as of March 23, 2026)
| Rank | BPB | Key technique |
|------|-----|--------------|
| #1 | **1.1194** | 11L/512d + LeakyReLU² + TTT + Parallel Muon + int6+LZMA |
| #2 | 1.1233 | 11L/512d + SmearGate + BigramHash + QAT + GPTQ-lite |
| Baseline | 1.2244 | Standard 11L/512d transformer |

## WHAT THE #1 SUBMISSION USES (our base fork)
**Architecture (GPT class, 27M params):**
- 11 transformer layers, 512d model_dim, 8 heads (GQA: 4 KV heads), head_dim=64
- MLP: 3× expansion with LeakyReLU(0.5)² activation
- Partial RoPE (only 16 of 64 head dims)
- U-Net skip connections (5 encoder + 6 decoder layers, learned skip weights)
- SmearGate: learned per-dim gate mixing current + previous token embedding
- BigramHashEmbedding: XOR hash of adjacent tokens → 2048-entry embedding → project to model_dim
- XSA (eXclude Self Attention) on last 4 layers: subtracts self-value projection
- ValueEmbedding on layers 9,10: re-injects token identity into value stream
- Logit softcap at 30.0 (tanh clamping)
- Tied embeddings (embedding = output projection)
- Learned per-dim attn_scale, mlp_scale, resid_mix (mixing x with initial embedding x0) per layer
- QK RMSNorm per head before RoPE
- ln_scale: 1/sqrt(layer_idx+1) normalization factor

**Optimizer:**
- **Parallel Muon** for weight banks (4 bank tensors): Newton-Schulz 5 orthogonalization, momentum=0.99, lr=0.025
  - 3-phase overlapped: async reduce-scatter → Adam steps on small params → wait RS + NS5 + async all-gather
  - No DDP at all — manual communication
- **AdamW** for embeddings (lr=0.035), scalars (lr=0.025)
- Gradient clipping at 0.3
- Momentum warmup: 0.92 → 0.99 over 1500 steps
- Weight decay: 0.04

**Training:**
- Batch: 786,432 tokens per step, seq_len=2048
- Wallclock-aware warmdown (LR → 0 at exactly 10 min)
- 20-step kernel warmup then FULL RESET of model+optimizer
- torch.compile(fullgraph=True)
- EMA (decay=0.997) applied to final weights
- SWA: snapshots every 50 steps when LR scale < 0.2

**Quantization:**
- **int6** per-row quantization for attention + MLP weights (clip range [-31,31])
  - Searches 5 clipping percentiles, picks lowest MSE
- **int8** for embeddings
- **float16** for small control tensors
- **LZMA** preset 6 compression
- Late QAT: int6 STE simulation activates when LR scale < 0.15

**Evaluation:**
- Sliding window with stride=64 for final BPB
- **TTT (Test-Time Training)**: legal score-first protocol — score each chunk, THEN train on it
  - SGD with momentum 0.9, lr=0.002, 3 epochs per chunk, 32K token chunks
  - Freezes first 2 blocks during TTT

**Weight storage:** All Q,K,V,O,MLP weights stored as 3D "bank" tensors for batched NS5:
- `qo_bank`: [2×11, 512, 512] (Q and O projections)
- `kv_bank`: [2×11, 256, 512] (K and V projections)
- `mlp_up_bank`: [11, 1536, 512]
- `mlp_down_bank`: [11, 512, 1536]

## OUR INNOVATION: DEPTH RECURRENCE + LoRA

Instead of 11 unique layers, we use **4 shared transformer blocks repeated 3 times = 12 effective layers**. This saves massive parameters, letting us go **wider (768d vs 512d)**.

**Key changes from #1:**
- Banks: [2×4, 768, 768] instead of [2×11, 512, 512]
- Per-position rank-8 LoRA adapters on Q and MLP_up (12 positions × 2 projections)
- 12 unique Block modules (own scales, mixes, gains) sharing 4 weight banks
- model_dim=768, num_heads=12, num_kv_heads=4

**Our model: 22.2M params (vs 27M original). Artifact ~10.5MB (vs ~15MB).**

## OUR EXPERIMENTAL RESULTS (local RTX 4070 SUPER, 500 steps, FineWeb real data)

### Ablation sweep (individual SOTA techniques, 11L/512d baseline):
| Config | Loss (500 steps) | Delta vs baseline |
|--------|-----------------|-------------------|
| FULL_SOTA (all techniques) | 4.0427 | **-0.1197** |
| +bigram_hash | 4.1124 | -0.0499 |
| +leaky_relu_sq | 4.1434 | -0.0190 |
| baseline_11L_3x | 4.1623 | — |
| +smeargate | 4.2114 | +0.0492 (hurt at 500 steps) |

### Depth recurrence sweep:
| Config | Params | Loss | Delta vs baseline | Artifact |
|--------|--------|------|-------------------|----------|
| rec_4x3_512_sota | 10.3M | 4.0944 | **-0.0984** | 5.5MB |
| rec_6x2_640_sota | 23.1M | 4.1065 | -0.0566 | 10.3MB |
| rec_5x3_640_sota | 19.5M | 4.1071 | -0.0559 | 8.5MB |
| rec_4x3_640_sota | 15.8M | 4.1227 | -0.0396 | 7.5MB |
| baseline_11L_3x | 26.5M | 4.1930 | — | 13.1MB |

**Key insight:** rec_4x3_512_sota wins at 500 steps. But wider models (640/768d) likely win at 14K+ steps (they're still converging).

## CONSTRAINTS AND REALITIES
1. **We haven't run on 8xH100 yet.** All experiments are local 500-step proxy tests.
2. **The #1 has been iterated for weeks.** We're building on their work, not starting from scratch.
3. **Budget: ~$50-75** for 2-3 hours of 8xH100 time.
4. **Flash Attention 3** is available on H100 (our code has SDPA fallback for local).
5. **torch.compile** with fullgraph=True is critical for throughput.
6. **The competition is open-source** — building on others' submissions is expected.

## WHAT WE WANT FROM YOU

Think outside the box. Consider EVERY possible improvement to maximize our BPB score. Areas to explore:

1. **Architecture improvements** that work within 16MB and 10 minutes:
   - Is 4×3 optimal or should we try 3×4, 5×3, 6×2?
   - Should LoRA rank be higher (16, 32)?
   - Should LoRA be on more projections (K, V, O, MLP_down)?
   - Mixture of Experts in MLP?
   - Different activation functions?
   - Attention alternatives (linear attention, sliding window)?

2. **Training improvements** within the 10-minute window:
   - Better learning rate schedule?
   - Progressive sequence length (start short, end long)?
   - Curriculum learning (easy→hard batches)?
   - Better batch size scheduling?
   - Different optimizer combinations?

3. **Quantization improvements** for smaller artifact:
   - Can we go to int4 for some weights?
   - Better compression than LZMA?
   - Mixed precision quantization strategies?
   - Pruning before quantization?

4. **Evaluation tricks** (legal under competition rules):
   - Better TTT parameters?
   - Ensemble of checkpoints within 16MB?
   - Better sliding window strategy?

5. **Novel ideas nobody has tried:**
   - Deep supervision (auxiliary prediction heads at repetition boundaries)
   - Stochastic depth during training (randomly skip reps)
   - Progressive widening (start narrow, expand mid-training)
   - Knowledge distillation from larger model during training
   - Adaptive computation (skip reps for "easy" tokens)

6. **What could go WRONG with depth recurrence at scale?**
   - Gradient flow issues with 3 passes through same weights?
   - Bank optimizer (Muon) may behave differently with shared banks?
   - LoRA may not provide enough specialization?

**Be specific. Give concrete suggestions with expected impact. Prioritize by risk/reward ratio.**
