"""Compare FIXED TTT vs CAUM Behavioral Regime TTT on real validation data."""
import sys, os, time, math, zlib, statistics
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
os.environ['DATA_PATH'] = './data/datasets/fineweb10B_sp1024'

from train_gpt_depthrecur import (
    GPT, Hyperparameters, load_data_shard, load_validation_tokens
)
from pathlib import Path

args = Hyperparameters()
device = 'cuda'

# Build model
model = GPT(
    vocab_size=1024, num_layers=12, model_dim=768, num_heads=12,
    num_kv_heads=4, mlp_mult=3, tie_embeddings=True,
    tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
    qk_gain_init=1.5, bigram_vocab_size=2048, bigram_dim=128,
    xsa_last_n=4, rope_dims=16, ln_scale=True,
    ve_enabled=True, ve_dim=128, ve_layers='10,11',
    num_unique_blocks=4, repeats=3, lora_rank=8,
    deep_sup_enabled=False,
).to(device).bfloat16()

print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')

# Quick train 300 steps so the model has some knowledge
shard = load_data_shard(Path(args.data_path) / 'fineweb_train_000000.bin')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
seq_len = 512
t0 = time.time()
for step in range(300):
    start = step * seq_len * 8
    tokens = shard[start:start + seq_len*8 + 1].to(dtype=torch.int64, device=device)
    x = tokens[:-1].reshape(8, seq_len)
    y = tokens[1:].reshape(8, seq_len)
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    optimizer.step()
print(f'Trained 300 steps in {time.time()-t0:.1f}s, final loss={loss.item():.4f}')

# Load validation data
val_tokens = load_validation_tokens(args.val_files, 512)
print(f'Val tokens: {val_tokens.numel():,}')

chunk_size = 16384
num_chunks = min(30, (val_tokens.numel() - 1) // chunk_size)
eval_seq_len = 512


def score_chunk(mdl, cs, ce):
    mdl.eval()
    total_nll = 0.0
    total_tokens = 0
    nll_values = []
    with torch.inference_mode():
        n_seqs = (ce - cs) // eval_seq_len
        for si in range(0, n_seqs, 8):
            be = min(si + 8, n_seqs)
            bs = be - si
            x_b = torch.zeros(bs, eval_seq_len, dtype=torch.int64, device=device)
            y_b = torch.zeros(bs, eval_seq_len, dtype=torch.int64, device=device)
            for i in range(bs):
                s = cs + (si + i) * eval_seq_len
                tok = val_tokens[s:s + eval_seq_len + 1].to(dtype=torch.int64, device=device)
                x_b[i] = tok[:-1]
                y_b[i] = tok[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = mdl.forward_logits(x_b)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_b.reshape(-1), reduction='none'
            ).reshape(bs, eval_seq_len)
            for i in range(bs):
                vals = nll[i].cpu().tolist()
                nll_values.extend(vals)
                total_nll += sum(vals)
                total_tokens += eval_seq_len
    return total_nll, total_tokens, nll_values


class RegimeClassifier:
    """CAUM Behavioral Regime Classifier — conservative, budget-saving.

    Key insight from experiments:
    - Over-training on hard chunks DAMAGES subsequent chunks (cascading effect)
    - TTT is sequential: every weight update propagates to all future chunks
    - The correct strategy: save compute on easy chunks, NEVER over-train on hard ones
    - Target: ≤ baseline total epochs, same or better BPB
    """
    def __init__(self, base_epochs=3):
        self.base_epochs = base_epochs
        self.nll_ema = None
        self.ema_alpha = 0.3

    def classify(self, nll_values, chunk_tokens_np, debug=False):
        if len(nll_values) < 10:
            return 'EXPAND', self.base_epochs, 1.0

        nll_mean = statistics.mean(nll_values)
        nll_std = statistics.stdev(nll_values)
        nll_cv = nll_std / max(nll_mean, 1e-6)
        zlib_ratio = len(zlib.compress(chunk_tokens_np.tobytes(), level=1)) / max(len(chunk_tokens_np.tobytes()), 1)
        mid = len(nll_values) // 2
        nll_trend = statistics.mean(nll_values[mid:]) - statistics.mean(nll_values[:mid])
        rel_trend = nll_trend / max(nll_mean, 1e-6)

        if self.nll_ema is None:
            self.nll_ema = nll_mean
        else:
            self.nll_ema = self.ema_alpha * nll_mean + (1 - self.ema_alpha) * self.nll_ema

        rel_diff = nll_mean / max(self.nll_ema, 1e-6)

        if debug:
            print(f'    [DEBUG] nll={nll_mean:.3f} cv={nll_cv:.3f} zlib={zlib_ratio:.4f} '
                  f'trend={rel_trend:+.4f} rel={rel_diff:.3f} ema={self.nll_ema:.3f}')

        # GRIND: chunk is easy relative to recent history
        # → Model already knows this. Reduce epochs to save compute.
        if rel_diff < 0.96:
            return 'GRIND', 2, 0.9

        # HARD: chunk is harder than recent. DON'T over-train.
        # Same epochs, slightly LOWER lr to prevent damage to future chunks.
        if rel_diff > 1.04:
            return 'HARD', self.base_epochs, 0.85

        # EXPAND: default
        return 'EXPAND', self.base_epochs, 1.0


def ttt_on_chunk(mdl, opt, cs, ce, epochs, lr_mult):
    if epochs <= 0:
        return
    mdl.train()
    for pg in opt.param_groups:
        pg['lr'] = 0.002 * lr_mult
    n_seqs = (ce - cs) // eval_seq_len
    for _ep in range(epochs):
        for si in range(0, n_seqs, 8):
            be = min(si + 8, n_seqs)
            s = cs + si * eval_seq_len
            e = cs + be * eval_seq_len + 1
            if e > val_tokens.numel():
                continue
            local = val_tokens[s:e].to(dtype=torch.int64, device=device)
            x = local[:-1].reshape(-1, eval_seq_len)
            y = local[1:].reshape(-1, eval_seq_len)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = mdl(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()


# Save model state for fair comparison
base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# ============ TEST A: FIXED 3 epochs ============
print('\n=== TEST A: FIXED 3 epochs per chunk ===')
model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
ttt_opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0)

fixed_nll = 0.0
fixed_tok = 0
t0 = time.time()
for ci in range(num_chunks):
    cs = ci * chunk_size
    ce = min((ci + 1) * chunk_size, val_tokens.numel() - 1)
    nll_sum, tok_cnt, _ = score_chunk(model, cs, ce)
    fixed_nll += nll_sum
    fixed_tok += tok_cnt
    if ci < num_chunks - 1:
        ttt_on_chunk(model, ttt_opt, cs, ce, epochs=3, lr_mult=1.0)
    if ci % 10 == 0:
        bpb_so_far = (fixed_nll / fixed_tok) / math.log(2.0)
        print(f'  chunk {ci}/{num_chunks}: running_bpb={bpb_so_far:.6f}')
fixed_time = time.time() - t0
fixed_bpb = (fixed_nll / fixed_tok) / math.log(2.0)
print(f'  FINAL BPB: {fixed_bpb:.6f} | Time: {fixed_time:.1f}s | Epochs: {3*num_chunks}')

# ============ TEST B: CAUM regime TTT ============
print('\n=== TEST B: CAUM Behavioral Regime TTT ===')
model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
ttt_opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0)

caum_nll = 0.0
caum_tok = 0
regime_counts = {'EXPAND': 0, 'GRIND': 0, 'HARD': 0}
total_epochs = 0
classifier = RegimeClassifier()
t0 = time.time()
for ci in range(num_chunks):
    cs = ci * chunk_size
    ce = min((ci + 1) * chunk_size, val_tokens.numel() - 1)
    nll_sum, tok_cnt, nll_vals = score_chunk(model, cs, ce)
    caum_nll += nll_sum
    caum_tok += tok_cnt
    # Classify
    chunk_np = val_tokens[cs:ce].numpy().astype(np.int16)
    regime, epochs, lr_mult = classifier.classify(nll_vals, chunk_np, debug=True)
    regime_counts[regime] += 1
    total_epochs += epochs
    if ci < num_chunks - 1:
        ttt_on_chunk(model, ttt_opt, cs, ce, epochs=epochs, lr_mult=lr_mult)
    bpb_so_far = (caum_nll / caum_tok) / math.log(2.0)
    print(f'  chunk {ci}/{num_chunks}: regime={regime} ep={epochs} lr={lr_mult:.1f} bpb={bpb_so_far:.6f}')
caum_time = time.time() - t0
caum_bpb = (caum_nll / caum_tok) / math.log(2.0)
print(f'  FINAL BPB: {caum_bpb:.6f} | Time: {caum_time:.1f}s | Epochs: {total_epochs}')
print(f'  Regimes: {dict(regime_counts)}')

# ============ COMPARISON ============
print('\n' + '='*60)
print('COMPARISON')
print('='*60)
delta = caum_bpb - fixed_bpb
winner = "CAUM WINS" if delta < 0 else "FIXED WINS" if delta > 0 else "TIE"
print(f'  Fixed 3-epoch BPB:  {fixed_bpb:.6f}  ({fixed_time:.1f}s, {3*num_chunks} epochs)')
print(f'  CAUM Regime BPB:    {caum_bpb:.6f}  ({caum_time:.1f}s, {total_epochs} epochs)')
print(f'  Delta BPB:          {delta:+.6f}  --> {winner}')
print(f'  Time saved:         {fixed_time - caum_time:+.1f}s')
print(f'  Epoch savings:      {3*num_chunks - total_epochs} fewer epochs')
