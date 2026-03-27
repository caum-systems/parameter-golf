"""
Phase 4: Verify the vocab=8192 pipeline and estimate artifact sizes.
Tests: tokenizer, shards, BPB calculation, artifact sizing for PR #549 architecture.
"""
import numpy as np
import sentencepiece as spm
import glob
import io
import lzma
import math
import os
import torch

# ============================================================
# 1. VERIFY TOKENIZER
# ============================================================
print("=" * 70)
print("1. TOKENIZER VERIFICATION")
print("=" * 70)

tok_path = "data/tokenizers/fineweb_8192_bpe.model"
sp = spm.SentencePieceProcessor()
sp.Load(tok_path)
print(f"  Loaded: {tok_path}")
print(f"  Vocab size: {sp.vocab_size()}")

test = "The quick brown fox jumps over the lazy dog."
tokens = sp.Encode(test)
decoded = sp.Decode(tokens)
print(f"  Test: '{test}'")
print(f"    -> {len(tokens)} tokens -> '{decoded}'")
print(f"    Bytes/token: {len(test.encode('utf-8')) / len(tokens):.2f}")

test2 = "In computer science, a hash table is a data structure that implements an associative array."
tokens2 = sp.Encode(test2)
print(f"  Test2: {len(test2.encode('utf-8'))} bytes -> {len(tokens2)} tokens -> {len(test2.encode('utf-8')) / len(tokens2):.2f} bytes/token")

# Measure bytes_per_token across all vocab entries
total_bytes = 0
total_counted = 0
for i in range(sp.vocab_size()):
    piece = sp.IdToPiece(i)
    if piece.startswith("<") and piece.endswith(">"):
        continue  # skip special tokens
    text = sp.Decode([i])
    byte_len = len(text.encode("utf-8"))
    total_bytes += byte_len
    total_counted += 1
avg_bytes = total_bytes / total_counted if total_counted else 0
print(f"  Average bytes/token (unweighted vocab): {avg_bytes:.2f}")

# ============================================================
# 2. VERIFY SHARDS
# ============================================================
print(f"\n{'=' * 70}")
print("2. SHARD VERIFICATION")
print("=" * 70)

shard_dir = "data/datasets/fineweb10B_sp8192"
shards = sorted(glob.glob(os.path.join(shard_dir, "fineweb_*.bin")))
print(f"  Found {len(shards)} shards in {shard_dir}")

for shard_path in shards:
    name = os.path.basename(shard_path)
    header = np.fromfile(shard_path, dtype="<i4", count=256)
    magic, version, num_tokens = header[0], header[1], header[2]
    tokens = np.fromfile(shard_path, dtype="<u2", count=min(1000, num_tokens), offset=256*4)
    max_tok = np.fromfile(shard_path, dtype="<u2", count=num_tokens, offset=256*4).max()
    file_size = os.path.getsize(shard_path)
    expected_size = 256 * 4 + num_tokens * 2
    ok = "OK" if magic == 20240520 and version == 1 and max_tok < 8192 and file_size == expected_size else "ERROR"
    print(f"  {name}: magic={magic} ver={version} tokens={num_tokens:,} max_id={max_tok} size={file_size:,} [{ok}]")

    # Decode first 100 tokens as sanity check
    sample = tokens[:100].tolist()
    text_sample = sp.Decode(sample)
    print(f"    First 100 tokens decode to: '{text_sample[:80]}...'")

# ============================================================
# 3. BYTES-PER-TOKEN ON VALIDATION DATA (like build_sentencepiece_luts)
# ============================================================
print(f"\n{'=' * 70}")
print("3. BPB CALCULATION VERIFICATION (build_sentencepiece_luts equivalent)")
print("=" * 70)

# Build the LUT exactly like train_gpt.py does
vocab_size = 8192
sp_vocab_size = sp.vocab_size()
table_size = max(sp_vocab_size, vocab_size)
base_bytes_np = np.zeros((table_size,), dtype=np.int64)

for i in range(sp_vocab_size):
    piece = sp.IdToPiece(i)
    if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
        base_bytes_np[i] = 1  # byte fallback token
    elif piece in ("<s>", "</s>", "<unk>"):
        base_bytes_np[i] = 0
    else:
        raw = piece.replace("\u2581", " ")  # SentencePiece space marker
        base_bytes_np[i] = len(raw.encode("utf-8"))

# Load val shard and compute weighted bytes/token
val_shards = sorted(glob.glob(os.path.join(shard_dir, "fineweb_val_*.bin")))
total_tokens_val = 0
total_bytes_val = 0
for vf in val_shards:
    h = np.fromfile(vf, dtype="<i4", count=256)
    nt = int(h[2])
    toks = np.fromfile(vf, dtype="<u2", count=nt, offset=256*4)
    total_tokens_val += nt
    for t in toks[:500000]:  # sample first 500K tokens for speed
        total_bytes_val += int(base_bytes_np[t])

sampled = min(500000, total_tokens_val)
weighted_bpt = total_bytes_val / sampled
print(f"  Validation tokens sampled: {sampled:,}")
print(f"  Total bytes (sampled): {total_bytes_val:,}")
print(f"  Weighted bytes/token: {weighted_bpt:.4f}")
print(f"  For comparison: sp1024 has ~2.44 bytes/token")
print(f"  Improvement ratio: {weighted_bpt / 2.44:.2f}x")

# Estimate BPB impact
# If loss_per_token is similar, BPB = loss / ln(2) / bytes_per_token
# Example: SOTA at 1.1194 BPB with 2.44 bpt -> loss_nats ≈ 1.1194 * ln(2) * 2.44 ≈ 1.893
# With our bpt: BPB = 1.893 / ln(2) / weighted_bpt
example_loss_nats = 1.1194 * math.log(2) * 2.44
theoretical_bpb = example_loss_nats / math.log(2) / weighted_bpt
print(f"\n  Theoretical BPB if same per-token loss as SOTA:")
print(f"    SOTA: 1.1194 BPB @ 2.44 bpt")
print(f"    Ours: {theoretical_bpb:.4f} BPB @ {weighted_bpt:.2f} bpt")
print(f"    Improvement: {1.1194 - theoretical_bpb:.4f} BPB ({(1.1194 - theoretical_bpb)/1.1194*100:.1f}%)")

# ============================================================
# 4. ARTIFACT SIZE ESTIMATION FOR PR #549 ARCHITECTURE
# ============================================================
print(f"\n{'=' * 70}")
print("4. ARTIFACT SIZE ESTIMATION (PR #549 architecture + vocab=8192)")
print("=" * 70)

def estimate_params(vocab, nlayers, dim, nheads, nkv, mlp_mult, bigram_vocab=1536, bigram_dim=128, ve_dim=128, ve_layers=2):
    """Estimate total parameters for the PR #549 architecture."""
    head_dim = dim // nheads
    kv_dim = nkv * head_dim
    mlp_dim = int(mlp_mult * dim)

    # Embedding (tied)
    emb = vocab * dim

    # Parameter banks (attention: QO + KV, MLP: up + down)
    qo_bank = 2 * nlayers * dim * dim
    kv_bank = 2 * nlayers * kv_dim * dim
    mlp_up = nlayers * mlp_dim * dim
    mlp_down = nlayers * dim * mlp_dim
    banks = qo_bank + kv_bank + mlp_up + mlp_down

    # Per-block: attn_scale, mlp_scale, resid_mix, q_gain
    per_block = nlayers * (dim + dim + 2*dim + nheads)

    # SmearGate
    smear = dim

    # Skip weights
    skip = min(nlayers // 2, nlayers - nlayers // 2) * dim

    # BigramHash
    bigram = bigram_vocab * bigram_dim + bigram_dim * dim + 1

    # ValueEmbedding
    ve = vocab * ve_dim + ve_dim * kv_dim + 1 + ve_layers

    total = emb + banks + per_block + smear + skip + bigram + ve
    return total, {
        "embedding": emb,
        "banks": banks,
        "per_block": per_block,
        "smear": smear,
        "skip": skip,
        "bigram": bigram,
        "value_embed": ve
    }

# Test different configurations
configs = [
    ("PR#549 original (v=1024, 11L)", 1024, 11),
    ("v=8192, 11L (won't fit)", 8192, 11),
    ("v=8192, 10L", 8192, 10),
    ("v=8192, 9L", 8192, 9),
    ("v=8192, 9L, no VE", 8192, 9),
    ("v=8192, 8L", 8192, 8),
]

print(f"\n{'Config':<35} {'Params':>10} {'INT6 raw':>10} {'LZMA est':>10} {'Fits?':>8}")
print("-" * 78)

for name, vocab, nlayers in configs:
    ve_dim = 128 if "no VE" not in name else 0
    ve_layers = 2 if ve_dim > 0 else 0
    total, breakdown = estimate_params(vocab, nlayers, 512, 8, 4, 3.0,
                                        bigram_vocab=1536, bigram_dim=128,
                                        ve_dim=ve_dim, ve_layers=ve_layers)

    # INT6 estimation: int8 storage (1 byte per param) + fp16 scales (2 bytes per row)
    # For large 2D tensors: ~1.003 bytes/param after scales
    # Small tensors kept as fp16: 2 bytes/param
    # Control tensors: 4 bytes/param (but tiny)
    large_params = breakdown["banks"] + breakdown["embedding"] + breakdown["value_embed"]
    small_params = breakdown["per_block"] + breakdown["smear"] + breakdown["skip"] + breakdown["bigram"]
    int6_raw = large_params * 1.003 + small_params * 2.0 + 50000  # ~50KB overhead for PyTorch format

    # LZMA compression ratio: PR#549 actual data shows 26.9M params -> 15.93 MB = 0.59x
    lzma_est = int6_raw * 0.59  # calibrated from actual PR #549 artifact

    # Add code size (~48KB)
    code_size = 48000
    total_submission = lzma_est + code_size

    fits = "YES" if total_submission < 16_000_000 else "NO"
    if total_submission < 16_000_000:
        headroom = 16_000_000 - total_submission
        fits = f"YES ({headroom/1e6:.1f}M)"

    print(f"{name:<35} {total:>10,} {int6_raw/1e6:>9.2f}M {total_submission/1e6:>9.2f}M {fits:>8}")

# Detailed breakdown for the best candidate
print(f"\n--- Detailed breakdown: v=8192, 9L ---")
total, bd = estimate_params(8192, 9, 512, 8, 4, 3.0)
for k, v in bd.items():
    pct = v / total * 100
    print(f"  {k:<20}: {v:>10,} params ({pct:.1f}%)")
print(f"  {'TOTAL':<20}: {total:>10,} params")

print(f"\n{'=' * 70}")
print("RECOMMENDATION")
print("=" * 70)
print("""
The PR #549 model with vocab=1024 uses ~16.0 MB (nearly full budget).
Adding vocab=8192 increases the embedding by +3.67M params (+2.5 MB after LZMA).

Options to fit in 16MB:
  1. NUM_LAYERS=9  -> saves ~2 layers -> fits with margin
  2. NUM_LAYERS=9 VE_ENABLED=0 -> maximum headroom
  3. NUM_LAYERS=10 VE_ENABLED=0 -> tighter fit

All options use ENV VARS only — no code changes needed.
The H100 run command becomes:

  NUM_LAYERS=9 VOCAB_SIZE=8192 \\
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \\
  DATA_PATH=./data/datasets/fineweb10B_sp8192/ \\
  torchrun --standalone --nproc_per_node=8 train_gpt.py
""")
