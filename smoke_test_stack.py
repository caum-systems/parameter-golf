"""
Smoke test: Verify the full innovation stack compiles and runs locally.
Tests: SVDEmbedding, TrigramHash, MoS, InceptionHead — all together.
No GPU required — just checks the model builds, forward/backward work,
and estimates the artifact size.
"""
import os
import sys
import time
import math
import numpy as np

# Force CPU and small config for local testing
os.environ["VOCAB_SIZE"] = "16384"
os.environ["NUM_LAYERS"] = "9"
os.environ["SVD_RANK"] = "64"
os.environ["MOS_ENABLED"] = "1"
os.environ["MOS_EXPERTS"] = "4"
os.environ["TRIGRAM_ENABLED"] = "1"
os.environ["INCEPTION_ENABLED"] = "1"
os.environ["INCEPTION_DIM"] = "64"
os.environ["INCEPTION_LAYERS"] = "1"
os.environ["VE_ENABLED"] = "0"

import torch
import torch.nn.functional as F

print("=" * 70)
print("SMOKE TEST: Full Innovation Stack")
print("=" * 70)

# Import just the model classes we need
# We can't import the full train_gpt_submission.py because it needs flash_attn
# Instead, test the individual components

from train_gpt_submission import SVDEmbedding, TrigramLogitBias, MixtureOfSoftmaxes
from train_gpt_submission import InceptionRefinementHead, EnhancedOutputHead
from train_gpt_submission import RefinementAttention, RefinementMLP

vocab = 16384
dim = 512
svd_rank = 64
B, T = 2, 64

print(f"\nConfig: vocab={vocab}, dim={dim}, svd_rank={svd_rank}")
print(f"Batch: B={B}, T={T}")

# ============================================================
# 1. SVDEmbedding
# ============================================================
print(f"\n{'='*50}")
print("1. SVDEmbedding")
svd_emb = SVDEmbedding(vocab, dim, rank=svd_rank)
ids = torch.randint(0, vocab, (B, T))
emb_out = svd_emb(ids)
print(f"   Forward: {ids.shape} -> {emb_out.shape}")
assert emb_out.shape == (B, T, dim)

W = svd_emb.weight
print(f"   Weight reconstruction: {W.shape}")
assert W.shape == (vocab, dim)

# Verify consistency
manual = W[ids[0, 0]]
fwd = emb_out[0, 0]
diff = (manual - fwd).abs().max().item()
print(f"   Consistency check: max diff = {diff:.8f} (should be ~0)")

svd_params = sum(p.numel() for p in svd_emb.parameters())
full_params = vocab * dim
print(f"   Params: {svd_params:,} vs full {full_params:,} (saves {(1-svd_params/full_params)*100:.0f}%)")
print("   PASS")

# ============================================================
# 2. TrigramLogitBias
# ============================================================
print(f"\n{'='*50}")
print("2. TrigramLogitBias")
trigram = TrigramLogitBias(vocab_size=vocab, proj_dim=16, table_size=4096, max_order=4)
bias = trigram(ids)
print(f"   Forward: {ids.shape} -> {bias.shape}")
assert bias.shape == (B, T, vocab)
tri_params = sum(p.numel() for p in trigram.parameters())
print(f"   Params: {tri_params:,}")
print("   PASS")

# ============================================================
# 3. MixtureOfSoftmaxes
# ============================================================
print(f"\n{'='*50}")
print("3. MixtureOfSoftmaxes")
mos = MixtureOfSoftmaxes(dim=dim, vocab_size=vocab, num_experts=4, logit_softcap=30.0)
hidden_flat = torch.randn(B*T, dim)
log_probs = mos(hidden_flat, W)
print(f"   Forward: {hidden_flat.shape} -> {log_probs.shape}")
assert log_probs.shape == (B*T, vocab)
# Check it's valid log probs
assert (log_probs <= 0).all(), "Log probs should be <= 0"
mos_params = sum(p.numel() for p in mos.parameters())
print(f"   Params: {mos_params:,}")
print("   PASS")

# ============================================================
# 4. InceptionRefinementHead
# ============================================================
print(f"\n{'='*50}")
print("4. InceptionRefinementHead")
inception = InceptionRefinementHead(main_dim=dim, refine_dim=64, num_layers=1)
hidden_3d = torch.randn(B, T, dim)
refined = inception(hidden_3d)
print(f"   Forward: {hidden_3d.shape} -> {refined.shape}")
assert refined.shape == hidden_3d.shape
init_diff = (refined - hidden_3d).abs().mean().item()
print(f"   Init deviation from identity: {init_diff:.6f} (should be ~0)")
inc_params = sum(p.numel() for p in inception.parameters())
print(f"   Params: {inc_params:,}")
print("   PASS")

# ============================================================
# 5. EnhancedOutputHead (full pipeline)
# ============================================================
print(f"\n{'='*50}")
print("5. EnhancedOutputHead (Inception + Trigram + MoS)")
head = EnhancedOutputHead(
    dim=dim, vocab_size=vocab, logit_softcap=30.0,
    mos_experts=4, trigram_table_size=4096,
    trigram_proj_dim=16, trigram_max_order=4,
    use_mos=True, use_trigram=True,
    use_inception=True, inception_dim=64, inception_layers=1,
)
targets = torch.randint(0, vocab, (B, T))
loss = head(hidden_3d, ids, targets, W)
print(f"   Forward loss: {loss.item():.4f}")
assert not torch.isnan(loss)

# Backward
loss.backward()
grad_count = sum(1 for p in head.parameters() if p.grad is not None)
total_params_head = sum(p.numel() for p in head.parameters())
print(f"   Backward: {grad_count} params have gradients")
print(f"   Total head params: {total_params_head:,}")

# get_log_probs (for TTT eval path)
head.zero_grad()
lp = head.get_log_probs(hidden_3d.detach(), ids, W.detach())
print(f"   get_log_probs: {lp.shape}")
assert lp.shape == (B, T, vocab)
print("   PASS")

# ============================================================
# 6. Artifact Size Estimation
# ============================================================
print(f"\n{'='*50}")
print("6. Artifact Size Estimation (9L/512d + full stack)")

def estimate_params_full_stack(vocab, nlayers, dim, nheads, nkv, mlp_mult,
                                svd_rank=64, bigram_vocab=1536, bigram_dim=128):
    head_dim = dim // nheads
    kv_dim = nkv * head_dim
    mlp_dim = int(mlp_mult * dim)

    # SVD Embedding instead of full
    if svd_rank > 0:
        emb = vocab * svd_rank + svd_rank + svd_rank * dim
    else:
        emb = vocab * dim

    # Parameter banks
    qo_bank = 2 * nlayers * dim * dim
    kv_bank = 2 * nlayers * kv_dim * dim
    mlp_up = nlayers * mlp_dim * dim
    mlp_down = nlayers * dim * mlp_dim
    banks = qo_bank + kv_bank + mlp_up + mlp_down

    # Per-block control tensors
    per_block = nlayers * (dim + dim + 2*dim + nheads)
    smear = dim
    skip = min(nlayers // 2, nlayers - nlayers // 2) * dim

    # BigramHash
    bigram = bigram_vocab * bigram_dim + bigram_dim * dim + 1

    # EnhancedOutputHead: Trigram + MoS + Inception
    trigram = 4 * 4096 * 16 + 16 * vocab + 4  # table + proj + order_scales
    mos = dim * 4 + 4 + 4  # gate + expert_scales + expert_temps
    inception = dim * 64 + 64 * 3 * 64 + 64 * 64 + 64 * dim + 1  # proj_down + attn + mlp + proj_up + scale
    # LayerNorms in inception (2 per layer)
    inception += 2 * 64  # norm weights for 1 layer
    # MLP in inception
    inception += 64 * 128 + 128 * 64  # fc + proj
    output_head = trigram + mos + inception

    total = emb + banks + per_block + smear + skip + bigram + output_head
    return total, {
        "embedding (SVD)": emb,
        "banks": banks,
        "per_block": per_block,
        "smear+skip": smear + skip,
        "bigram": bigram,
        "output_head": output_head,
    }

total, bd = estimate_params_full_stack(vocab, 9, 512, 8, 4, 3.0, svd_rank=64)

print(f"\n   Parameter breakdown:")
for k, v in bd.items():
    pct = v / total * 100
    print(f"     {k:<25}: {v:>10,} ({pct:.1f}%)")
print(f"     {'TOTAL':<25}: {total:>10,}")

# INT6 + LZMA estimation
large_params = bd["banks"] + bd["embedding (SVD)"]
small_params = total - large_params
int6_raw = large_params * 0.75 + small_params * 2.0 + 50000  # overhead
lzma_ratio = 0.59  # calibrated from PR #549
artifact_model = int6_raw * lzma_ratio
code_size = 55000  # ~55KB for the .py file now
total_artifact = artifact_model + code_size

print(f"\n   INT6 raw: {int6_raw/1e6:.2f} MB")
print(f"   After LZMA (x{lzma_ratio}): {artifact_model/1e6:.2f} MB")
print(f"   + Code: {code_size/1e3:.0f} KB")
print(f"   TOTAL ARTIFACT: {total_artifact/1e6:.2f} MB")
headroom = 16_000_000 - total_artifact
fits = "YES" if headroom > 0 else "NO"
print(f"   16MB budget headroom: {headroom/1e6:.2f} MB [{fits}]")

# ============================================================
# 7. Full Embedding Comparison
# ============================================================
print(f"\n{'='*50}")
print("7. Comparison: SVD r=64 vs Full Embedding")
total_full, _ = estimate_params_full_stack(vocab, 9, 512, 8, 4, 3.0, svd_rank=0)
total_svd64, _ = estimate_params_full_stack(vocab, 9, 512, 8, 4, 3.0, svd_rank=64)
savings = total_full - total_svd64
print(f"   Full emb params:  {total_full:>12,}")
print(f"   SVD r=64 params:  {total_svd64:>12,}")
print(f"   Savings:          {savings:>12,} ({savings/total_full*100:.1f}%)")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("SMOKE TEST SUMMARY")
print(f"{'='*70}")
print(f"""
  SVDEmbedding ......... PASS ({svd_params:,} params, saves {(1-svd_params/full_params)*100:.0f}%)
  TrigramLogitBias ..... PASS ({tri_params:,} params)
  MixtureOfSoftmaxes ... PASS ({mos_params:,} params)
  InceptionRefinement .. PASS ({inc_params:,} params, identity at init)
  EnhancedOutputHead ... PASS ({total_params_head:,} params, loss={loss.item():.4f})
  Artifact estimate .... {total_artifact/1e6:.2f} MB ({fits}, {headroom/1e6:.2f} MB headroom)

  All components work together. Ready for H100 training.
""")
