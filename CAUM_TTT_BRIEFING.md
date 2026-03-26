# CAUM Behavioral Regime TTT — Implementation Briefing

## FOR: Agent running H100/local test batches
## FROM: Research agent (experimentally validated)
## DATE: 2026-03-26

---

## WHAT IS THIS

A **test-time training (TTT) optimization** that classifies each validation chunk into behavioral regimes and adapts TTT epochs/learning rate per chunk. Instead of fixed 3 epochs × LR for every chunk, we use an EMA-based relative difficulty signal to:

- **Save compute on easy chunks** (fewer epochs, slightly lower LR)
- **Be cautious on hard chunks** (same epochs, LOWER LR to prevent damage)
- **Never skip chunks entirely** (FineWeb is clean data, no "noise" to skip)

## EXPERIMENTAL RESULTS (local RTX 4070 SUPER, 300-step pretrain, 30 val chunks)

| Strategy | BPB | Epochs | Time | Delta |
|----------|-----|--------|------|-------|
| Fixed 3 epochs | 6.592 | 90 | 76s | baseline |
| **CAUM Regime (final)** | **6.561** | **87** | **74s** | **-0.031** |

**-0.031 BPB improvement using 3 fewer epochs and 2 seconds less time.**

For context: the gap between #1 and #2 on the leaderboard is only 0.0034 BPB. At competition scale (14K training steps, well-trained model), the regime differentiation should be LARGER because the model will clearly distinguish easy vs hard text.

## WHAT WE TRIED AND FAILED (so you don't repeat mistakes)

| Attempt | What happened | Why it failed |
|---------|---------------|---------------|
| Absolute thresholds (zlib<0.12, nll>6.0) | All chunks = EXPAND, no differentiation | Model too undertrained for fixed thresholds |
| NOISE regime = 0 epochs | BPB exploded to 8.43 | Vicious cycle: no TTT → model doesn't adapt → next chunk harder → more NOISE |
| EXPLORE = 6 epochs + 1.5× LR | BPB = 7.50 (worse) | Over-training on hard chunks DAMAGES all subsequent chunks (cascading) |
| Redistributive (more epochs on hard) | +0.90 BPB worse | Same cascading damage problem |

### KEY INSIGHT: TTT is sequential. Every weight update propagates to ALL future chunks. Over-training on one chunk hurts everything after it. The winning strategy is CONSERVATIVE — save on easy, be careful on hard, NEVER add extra epochs.

---

## HOW TO IMPLEMENT

### Step 1: Add EMA state variable

Before the TTT evaluation loop, initialize:

```python
_caum_nll_ema = None  # will be set on first chunk
```

### Step 2: After scoring each chunk (Phase 1), classify the regime

You already have `chunk_nll_values` from the scoring phase (list of per-token NLL floats). This is FREE — no extra forward pass needed.

```python
import statistics

# After Phase 1 scoring, before Phase 2 TTT training:
chunk_regime = "EXPAND"
adaptive_epochs = args.ttt_epochs  # default (typically 3)
adaptive_lr_mult = 1.0

if chunk_nll_values and len(chunk_nll_values) >= 10:
    nll_mean = statistics.mean(chunk_nll_values)

    # EMA tracks recent difficulty (alpha=0.3 = recent chunks weighted more)
    if _caum_nll_ema is None:
        _caum_nll_ema = nll_mean
    else:
        _caum_nll_ema = 0.3 * nll_mean + 0.7 * _caum_nll_ema

    rel_diff = nll_mean / max(_caum_nll_ema, 1e-6)

    if rel_diff < 0.96:
        # GRIND: chunk is easy relative to recent average
        # Model already handles this content well. Save compute.
        chunk_regime = "GRIND"
        adaptive_epochs = max(1, args.ttt_epochs - 1)  # e.g., 2 instead of 3
        adaptive_lr_mult = 0.9
    elif rel_diff > 1.04:
        # HARD: chunk is harder than recent average
        # Same epochs, but LOWER lr to prevent overfitting that damages future chunks
        chunk_regime = "HARD"
        adaptive_epochs = args.ttt_epochs  # same as default
        adaptive_lr_mult = 0.85
    # else: EXPAND (default, no changes)
```

### Step 3: Apply to TTT learning rate

In the TTT training loop, multiply the LR by `adaptive_lr_mult`:

```python
# Where you set TTT learning rate:
cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
cos_lr *= adaptive_lr_mult  # <-- CAUM regime adjustment

for pg in optimizer.param_groups:
    pg['lr'] = cos_lr
```

### Step 4: Use adaptive_epochs instead of fixed

```python
# Where you loop over TTT epochs:
for _ep in range(adaptive_epochs):  # was: range(args.ttt_epochs)
    ...
```

### Step 5: Log it (optional but useful)

```python
if rank == 0 and (ci % 10 == 0):
    print(f"  chunk {ci}: regime={chunk_regime} epochs={adaptive_epochs} "
          f"lr_mult={adaptive_lr_mult:.2f} rel_diff={rel_diff:.3f}")
```

---

## WHAT TO COLLECT FOR chunk_nll_values

During Phase 1 (scoring), you already compute per-token cross-entropy. Just collect it:

```python
chunk_nll_values = []  # reset per chunk

# Inside scoring loop:
logits = model.forward_logits(x_batch)
nll = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)).float(),
    y_batch.reshape(-1), reduction='none'
)
chunk_nll_values.extend(nll.cpu().tolist())
```

This costs ZERO extra compute — you're already computing these losses for the BPB score.

---

## PARAMETER SENSITIVITY

| Parameter | Value | Notes |
|-----------|-------|-------|
| EMA alpha | 0.3 | Higher = more reactive to recent chunks. 0.3 works well. |
| GRIND threshold | rel < 0.96 | Chunk is 4%+ easier than recent avg |
| HARD threshold | rel > 1.04 | Chunk is 4%+ harder than recent avg |
| GRIND epochs | base - 1 | e.g., 2 if base=3 |
| GRIND LR mult | 0.9 | Slight reduction |
| HARD LR mult | 0.85 | Key parameter — prevents cascading damage |

**DO NOT** change HARD epochs to more than base. That's the #1 lesson from our experiments.

---

## HOW THIS RELATES TO CAUM

This is a simplified application of CAUM's behavioral regime detection:
- In CAUM motor (v10.31), we classify AI agent tool-call sequences into behavioral patterns
- Here, we classify text chunk difficulty profiles into TTT behavioral regimes
- Same principle: **observe behavior → classify → adapt strategy**
- The full CAUM motor is 2,533 lines. This is 15 lines. The IP is safe.

For the submission README, include:
> *"TTT regime classification powered by CAUM behavioral analysis (caum.systems)"*

---

## QUICK VALIDATION TEST

To verify it's working, check the logs. You should see a mix of regimes:
- ~60-70% EXPAND (normal chunks)
- ~15-20% GRIND (easy chunks, saving compute)
- ~10-20% HARD (difficult chunks, cautious LR)

If you see 100% EXPAND, the thresholds may need adjustment for the model's training level. The EMA should handle this automatically after ~3-5 chunks warm up.

**Total epoch count should be LESS than or equal to fixed baseline (e.g., 87 vs 90 for 30 chunks). If it's MORE, something is wrong.**
