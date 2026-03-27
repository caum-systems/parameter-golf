"""
VOCAB SCALING EXPERIMENT: Find the Sweet Spot
==============================================

Question: At what vocab size does bytes_per_token cross 3.5+?

We test: 8K, 16K, 32K, 50K, 65K vocab sizes
For each: train tokenizer, measure bytes/token, estimate artifact size

The key tradeoff:
- Bigger vocab -> higher bytes/token (good for BPB denominator)
- Bigger vocab -> bigger embedding table (bad for artifact size)
- SVD compression on embeddings -> can we have both?

INSTRUCTIONS FOR CLAUDE CODE:
1. This needs the FineWeb text file from Phase 1 
   (fineweb_raw_text_for_tokenizer.txt or equivalent)
2. Run: python3 vocab_scaling_experiment.py
3. Takes ~20-30 minutes (training 5 tokenizers)
4. Report the results table at the end
"""

import os
import sys
import math
import time
import json

# Try to import sentencepiece
try:
    import sentencepiece as spm
except ImportError:
    print("ERROR: pip install sentencepiece")
    sys.exit(1)

OUTPUT_DIR = "./vocab_scaling_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

# Path to FineWeb raw text (from Phase 1)
# Try multiple possible locations
TEXT_FILE_CANDIDATES = [
    "fineweb_raw_text_for_tokenizer.txt",
    "tokenizer_experiment_output/fineweb_sample.txt",
    "../fineweb_raw_text_for_tokenizer.txt",
    os.path.expanduser("~/parameter-golf/fineweb_raw_text_for_tokenizer.txt"),
]

TEXT_FILE = None
for candidate in TEXT_FILE_CANDIDATES:
    if os.path.exists(candidate):
        TEXT_FILE = candidate
        break

if TEXT_FILE is None:
    print("ERROR: Cannot find FineWeb text file.")
    print("Expected one of:", TEXT_FILE_CANDIDATES)
    print("\nRun Phase 1 first to generate the text file.")
    sys.exit(1)

print(f"Using text file: {TEXT_FILE}")
print(f"File size: {os.path.getsize(TEXT_FILE) / 1e6:.0f} MB")

# Vocab sizes to test
VOCAB_SIZES = [8192, 16384, 32768, 50000, 65536]

# Model configs for artifact estimation
MODEL_DIM = 512
NUM_LAYERS = 9  # like current submission plan
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 3

# ============================================================
# STEP 1: Train tokenizers at each vocab size
# ============================================================

def train_tokenizer(text_path, vocab_size, output_dir):
    """Train SentencePiece BPE tokenizer."""
    prefix = os.path.join(output_dir, f"sp_{vocab_size}")
    model_path = f"{prefix}.model"
    
    if os.path.exists(model_path):
        print(f"  [{vocab_size}] Already exists, skipping training")
        return model_path
    
    print(f"  [{vocab_size}] Training SentencePiece BPE...")
    t0 = time.time()
    
    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule_name="identity",
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=True,
        max_sentence_length=16384,
        input_sentence_size=5000000,
        shuffle_input_sentence=True,
    )
    
    t1 = time.time()
    print(f"  [{vocab_size}] Done in {t1-t0:.0f}s")
    return model_path


print("\n" + "=" * 70)
print("STEP 1: Training tokenizers")
print("=" * 70)

tokenizer_paths = {}
for vocab in VOCAB_SIZES:
    tokenizer_paths[vocab] = train_tokenizer(TEXT_FILE, vocab, OUTPUT_DIR)


# ============================================================
# STEP 2: Measure bytes-per-token for each
# ============================================================

def measure_bytes_per_token(tokenizer_path, text_path, max_chars=10_000_000):
    """Measure actual bytes-per-token on real text."""
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    
    total_bytes = 0
    total_tokens = 0
    doc_count = 0
    
    with open(text_path, "r", encoding="utf-8") as f:
        chars_read = 0
        for line in f:
            if chars_read > max_chars:
                break
            text = line.strip()
            if len(text) < 20:
                continue
            
            text_bytes = len(text.encode("utf-8"))
            token_ids = sp.Encode(text)
            num_tokens = len(token_ids)
            
            total_bytes += text_bytes
            total_tokens += num_tokens
            doc_count += 1
            chars_read += len(text)
    
    bpt = total_bytes / total_tokens if total_tokens > 0 else 0
    
    return {
        'vocab_size': sp.vocab_size(),
        'bytes_per_token': bpt,
        'total_bytes': total_bytes,
        'total_tokens': total_tokens,
        'docs': doc_count,
    }


print("\n" + "=" * 70)
print("STEP 2: Measuring bytes-per-token")
print("=" * 70)

measurements = {}
for vocab, tok_path in sorted(tokenizer_paths.items()):
    print(f"\n  Measuring vocab={vocab}...")
    m = measure_bytes_per_token(tok_path, TEXT_FILE)
    measurements[vocab] = m
    print(f"    bytes/token = {m['bytes_per_token']:.3f}")
    print(f"    measured on {m['docs']:,} docs, {m['total_tokens']:,} tokens")


# ============================================================
# STEP 3: Estimate artifact sizes with different embedding strategies
# ============================================================

def estimate_artifact(vocab_size, model_dim, num_layers, num_heads, 
                      num_kv_heads, mlp_mult, emb_strategy='full'):
    """
    Estimate compressed artifact size.
    emb_strategy: 'full', 'svd_r32', 'svd_r64', 'svd_r128'
    """
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    mlp_dim = mlp_mult * model_dim
    
    # Transformer body params
    per_layer = (
        model_dim * model_dim +      # Q
        model_dim * model_dim +      # O  
        kv_dim * model_dim +         # K
        kv_dim * model_dim +         # V
        mlp_dim * model_dim +        # MLP up
        model_dim * mlp_dim +        # MLP down
        model_dim * 4 +              # scalars (attn_scale, mlp_scale, resid_mix)
        num_heads                    # q_gain
    )
    body_params = per_layer * num_layers
    
    # Embedding params
    if emb_strategy == 'full':
        emb_params = vocab_size * model_dim
    elif emb_strategy.startswith('svd_r'):
        rank = int(emb_strategy.split('r')[1])
        # U: [vocab, rank] + S: [rank] + V: [rank, dim]
        emb_params = vocab_size * rank + rank + rank * model_dim
    else:
        emb_params = vocab_size * model_dim
    
    # BigramHash (if used)
    bigram_params = 1536 * model_dim  # reduced bigram vocab
    
    # SmearGate
    smear_params = model_dim
    
    total_params = body_params + emb_params + bigram_params + smear_params
    
    # Compression estimate: int6 + LZMA
    # From our data: ~0.68 bytes per param after int6+LZMA
    # But embeddings compress differently than weights
    body_compressed = body_params * 0.68
    
    if emb_strategy == 'full':
        emb_compressed = emb_params * 0.72  # embeddings compress slightly worse
    else:
        # SVD factors are denser, compress less well
        emb_compressed = emb_params * 0.75
    
    other_compressed = (bigram_params + smear_params) * 0.70
    
    total_compressed = body_compressed + emb_compressed + other_compressed
    
    # Code size (~45KB)
    code_bytes = 45_000
    
    artifact = total_compressed + code_bytes
    
    return {
        'total_params': total_params,
        'body_params': body_params,
        'emb_params': emb_params,
        'emb_strategy': emb_strategy,
        'estimated_artifact_bytes': int(artifact),
        'fits_16mb': artifact < 16_000_000,
        'headroom_bytes': int(16_000_000 - artifact),
    }


print("\n" + "=" * 70)
print("STEP 3: Artifact size estimates")
print("=" * 70)

print(f"\nBase config: {NUM_LAYERS}L/{MODEL_DIM}d, {NUM_HEADS}h/{NUM_KV_HEADS}kv, MLP {MLP_MULT}x")

all_configs = []

for vocab in VOCAB_SIZES:
    for emb_strat in ['full', 'svd_r32', 'svd_r64', 'svd_r128']:
        est = estimate_artifact(vocab, MODEL_DIM, NUM_LAYERS, NUM_HEADS,
                               NUM_KV_HEADS, MLP_MULT, emb_strat)
        est['vocab'] = vocab
        est['bpt'] = measurements[vocab]['bytes_per_token']
        all_configs.append(est)

print(f"\n{'Vocab':>7} | {'Emb Strategy':>12} | {'Emb Params':>10} | {'Artifact':>8} | {'Headroom':>8} | {'Fits':>5} | {'B/T':>5}")
print("-" * 80)

for c in all_configs:
    fits = "YES" if c['fits_16mb'] else "NO"
    print(f"{c['vocab']:>7} | {c['emb_strategy']:>12} | {c['emb_params']:>10,} | "
          f"{c['estimated_artifact_bytes']/1e6:>6.2f}MB | "
          f"{c['headroom_bytes']/1e6:>6.2f}MB | {fits:>5} | "
          f"{c['bpt']:>5.2f}")


# ============================================================
# STEP 4: BPB projections with real bytes_per_token
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: BPB Projections (with REAL bytes-per-token)")
print("=" * 70)

# Reference
REF_LOSS = 1.90  # SOTA #1 achieves this (gives BPB 1.12 with bpt=2.44)
REF_BPB = 1.1194
REF_BPT = 2.44
REF_VOCAB = 1024

# The SOTA loss of 1.90 is with all optimizations. 
# With vocab change, the loss changes.

print(f"\nReference: SOTA loss={REF_LOSS}, BPB={REF_BPB}, bytes/token={REF_BPT}")
print(f"\nProjections at different loss penalty assumptions:\n")

for penalty_per_doubling in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    print(f"  Penalty = {penalty_per_doubling} nats per vocab doubling:")
    print(f"  {'Vocab':>7} | {'B/T':>5} | {'Loss':>6} | {'BPB':>7} | {'ΔBPB':>7} | {'Fits w/SVD64':>12}")
    print(f"  {'-'*65}")
    
    for vocab in [1024] + VOCAB_SIZES:
        if vocab == 1024:
            bpt = REF_BPT
        else:
            bpt = measurements[vocab]['bytes_per_token']
        
        doublings = math.log2(vocab / REF_VOCAB)
        est_loss = REF_LOSS + penalty_per_doubling * doublings
        est_bpb = (est_loss / math.log(2)) / bpt
        delta = est_bpb - REF_BPB
        
        # Check if it fits with SVD r64
        if vocab <= 1024:
            fits = "baseline"
        else:
            est = estimate_artifact(vocab, MODEL_DIM, NUM_LAYERS, NUM_HEADS,
                                   NUM_KV_HEADS, MLP_MULT, 'svd_r64')
            fits = "YES" if est['fits_16mb'] else "NO"
        
        marker = " <--" if delta < -0.02 and fits == "YES" else ""
        print(f"  {vocab:>7} | {bpt:>5.2f} | {est_loss:>6.3f} | {est_bpb:>7.4f} | {delta:>+7.4f} | {fits:>12}{marker}")
    
    print()


# ============================================================
# STEP 5: Find the optimal configuration
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: OPTIMAL CONFIGURATION FINDER")
print("=" * 70)

print("""
For each vocab size that FITS in 16MB (with SVD embedding),
compute the BPB at different penalty assumptions.
Highlight the BEST vocab for each penalty level.
""")

best_per_penalty = {}

for penalty in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    best_vocab = 1024
    best_bpb = REF_BPB
    
    for vocab in VOCAB_SIZES:
        # Check if fits
        est = estimate_artifact(vocab, MODEL_DIM, NUM_LAYERS, NUM_HEADS,
                               NUM_KV_HEADS, MLP_MULT, 'svd_r64')
        if not est['fits_16mb']:
            continue
        
        bpt = measurements[vocab]['bytes_per_token']
        doublings = math.log2(vocab / REF_VOCAB)
        est_loss = REF_LOSS + penalty * doublings
        est_bpb = (est_loss / math.log(2)) / bpt
        
        if est_bpb < best_bpb:
            best_bpb = est_bpb
            best_vocab = vocab
    
    delta = best_bpb - REF_BPB
    best_per_penalty[penalty] = (best_vocab, best_bpb, delta)
    
    print(f"  Penalty {penalty:.2f}: Best vocab = {best_vocab:>6}, "
          f"BPB = {best_bpb:.4f} ({delta:+.4f} vs SOTA)")


# ============================================================
# STEP 6: Summary and recommendation
# ============================================================

print(f"\n\n{'='*70}")
print("SUMMARY AND RECOMMENDATION")
print(f"{'='*70}")

# Find the vocab size that wins across the most penalty levels
from collections import Counter
vocab_wins = Counter(v[0] for v in best_per_penalty.values())
most_robust_vocab = vocab_wins.most_common(1)[0][0]

print(f"""
BYTES-PER-TOKEN MEASUREMENTS (on 1GB FineWeb):
""")

for vocab in [1024] + VOCAB_SIZES:
    if vocab == 1024:
        bpt = REF_BPT
    else:
        bpt = measurements[vocab]['bytes_per_token']
    ratio = bpt / REF_BPT
    print(f"  vocab={vocab:>6}: {bpt:.3f} bytes/token ({ratio:.2f}x vs 1024)")

print(f"""

ARTIFACT SIZES (9L/512d, SVD rank 64 embeddings):
""")

for vocab in VOCAB_SIZES:
    est = estimate_artifact(vocab, MODEL_DIM, NUM_LAYERS, NUM_HEADS,
                           NUM_KV_HEADS, MLP_MULT, 'svd_r64')
    fits = "YES" if est['fits_16mb'] else "NO"
    print(f"  vocab={vocab:>6}: {est['estimated_artifact_bytes']/1e6:.2f}MB "
          f"(headroom: {est['headroom_bytes']/1e6:.2f}MB) {fits}")

print(f"""

MOST ROBUST VOCAB SIZE: {most_robust_vocab}
  Wins at {vocab_wins[most_robust_vocab]}/{len(best_per_penalty)} penalty levels

  bytes/token: {measurements.get(most_robust_vocab, {}).get('bytes_per_token', REF_BPT):.3f}
  Projected BPB improvement: {best_per_penalty[0.15][2]:+.4f} to {best_per_penalty[0.05][2]:+.4f}

DECISION FRAMEWORK:
  If penalty < 0.15 per doubling -> bigger vocab wins big
  If penalty 0.15-0.25 -> marginal, depends on exact number
  If penalty > 0.25 -> stick with 1024

  The ONLY way to know the real penalty is to train on H100.
  Cost: ~$8 per run.

NEXT STEPS:
  1. If most robust vocab > 8192: retrain tokenizer at that size
  2. Implement SVD embedding in train_gpt.py (if needed for that vocab)
  3. Run H100 with the optimal config
  4. Compare BPB vs baseline run
""")

# Save results
results = {
    'measurements': {str(k): v for k, v in measurements.items()},
    'best_per_penalty': {str(k): {'vocab': v[0], 'bpb': v[1], 'delta': v[2]} 
                         for k, v in best_per_penalty.items()},
    'most_robust_vocab': most_robust_vocab,
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
}

results_path = os.path.join(OUTPUT_DIR, "vocab_scaling_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {results_path}")
