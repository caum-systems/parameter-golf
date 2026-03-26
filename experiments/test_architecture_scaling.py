"""Test architecture scaling: use the 7.5MB headroom (8.48MB current vs 16MB budget).

Hypothesis: More parameters = lower BPB. We're leaving 47% of our budget unused.

Configs tested (all with VR=True, DS=True, LoRA=16):
  A) CURRENT:     4 blocks × 3 reps, dim=768, heads=12, kv=4  (~23M params, ~8.5MB)
  B) WIDER:       4 blocks × 3 reps, dim=896, heads=14, kv=4  (~30M params, ~11MB est.)
  C) MORE_BLOCKS: 6 blocks × 2 reps, dim=768, heads=12, kv=4  (~26M params, ~10MB est.)
  D) WIDER+RANK:  4 blocks × 3 reps, dim=896, heads=14, kv=4, rank=24 (~32M, ~12MB est.)
  E) MEGA_WIDE:   4 blocks × 3 reps, dim=1024, heads=16, kv=4 (~38M, ~14MB est.)
"""
import sys, os, time, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
os.environ['DATA_PATH'] = './data/datasets/fineweb10B_sp1024'

from train_gpt_depthrecur import GPT, Hyperparameters, load_data_shard, load_validation_tokens
from pathlib import Path

args = Hyperparameters()
device = 'cuda'
seq_len = 512
steps = 2000
batch = 8

shard = load_data_shard(Path(args.data_path) / 'fineweb_train_000000.bin')
val_tokens = load_validation_tokens(args.val_files, seq_len)
print(f'Data loaded. {val_tokens.numel():,} val tokens')


CONFIGS = {
    'A_CURRENT': dict(
        vocab_size=1024, num_layers=12, model_dim=768, num_heads=12,
        num_kv_heads=4, mlp_mult=3, num_unique_blocks=4, repeats=3,
        lora_rank=16, value_residual=True, deep_sup_enabled=True,
    ),
    'B_WIDER_896': dict(
        vocab_size=1024, num_layers=12, model_dim=896, num_heads=14,
        num_kv_heads=4, mlp_mult=3, num_unique_blocks=4, repeats=3,
        lora_rank=16, value_residual=True, deep_sup_enabled=True,
    ),
    'C_6BLOCKS': dict(
        vocab_size=1024, num_layers=12, model_dim=768, num_heads=12,
        num_kv_heads=4, mlp_mult=3, num_unique_blocks=6, repeats=2,
        lora_rank=16, value_residual=True, deep_sup_enabled=True,
    ),
    'D_WIDER_RANK24': dict(
        vocab_size=1024, num_layers=12, model_dim=896, num_heads=14,
        num_kv_heads=4, mlp_mult=3, num_unique_blocks=4, repeats=3,
        lora_rank=24, value_residual=True, deep_sup_enabled=True,
    ),
    'E_MEGA_1024': dict(
        vocab_size=1024, num_layers=12, model_dim=1024, num_heads=16,
        num_kv_heads=4, mlp_mult=3, num_unique_blocks=4, repeats=3,
        lora_rank=16, value_residual=True, deep_sup_enabled=True,
    ),
}


def make_model(cfg):
    return GPT(
        tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0,
        rope_base=10000.0, qk_gain_init=1.5, bigram_vocab_size=2048,
        bigram_dim=128, xsa_last_n=4, rope_dims=16, ln_scale=True,
        ve_enabled=True, ve_dim=128, ve_layers='10,11',
        deep_sup_weight=0.1, gated_attention=False, mtp_num_heads=0,
        **cfg,
    ).to(device).bfloat16()


def train_and_eval(model, label):
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    t0 = time.time()
    for step in range(steps):
        start = step * seq_len * batch
        tokens = shard[start:start + seq_len * batch + 1].to(dtype=torch.int64, device=device)
        x = tokens[:-1].reshape(batch, seq_len)
        y = tokens[1:].reshape(batch, seq_len)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            ms_per_step = elapsed / (step + 1) * 1000
            print(f'  [{label}] step={step+1} loss={loss.item():.4f} ms/step={ms_per_step:.0f}')
    train_time = time.time() - t0

    # Eval
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_eval = min(80, (val_tokens.numel() - 1) // seq_len)
    with torch.inference_mode():
        for si in range(0, n_eval, batch):
            be = min(si + batch, n_eval)
            bs = be - si
            x_b = torch.zeros(bs, seq_len, dtype=torch.int64, device=device)
            y_b = torch.zeros(bs, seq_len, dtype=torch.int64, device=device)
            for i in range(bs):
                s = (si + i) * seq_len
                tok = val_tokens[s:s + seq_len + 1].to(dtype=torch.int64, device=device)
                x_b[i] = tok[:-1]
                y_b[i] = tok[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model.forward_logits(x_b)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_b.reshape(-1), reduction='none'
            )
            total_nll += nll.sum().item()
            total_tokens += bs * seq_len
    val_loss = total_nll / total_tokens
    val_bpb = val_loss / math.log(2.0)
    print(f'  [{label}] params={n_params:,} val_bpb={val_bpb:.4f} time={train_time:.1f}s')
    return val_bpb, n_params, train_time


results = {}
for name, cfg in CONFIGS.items():
    print(f'\n{"="*60}')
    print(f'=== {name} ===')
    print(f'{"="*60}')
    torch.manual_seed(42)
    try:
        model = make_model(cfg)
        bpb, n_params, t = train_and_eval(model, name)
        results[name] = (bpb, n_params, t)
    except Exception as e:
        print(f'  FAILED: {e}')
        results[name] = (None, None, None)
    if 'model' in dir():
        del model
    torch.cuda.empty_cache()

print(f'\n{"="*60}')
print('ARCHITECTURE SCALING COMPARISON (2000 steps, lower BPB = better)')
print(f'{"="*60}')
baseline_bpb = results.get('A_CURRENT', (None,))[0]
for name in CONFIGS:
    if results[name][0] is None:
        print(f'  {name:20s}  FAILED')
        continue
    bpb, params, t = results[name]
    delta = bpb - baseline_bpb if baseline_bpb else 0
    est_mb = params * 0.75 / 1024 / 1024  # rough: ~6 bits/param avg with int6/int8 mix
    marker = ' <-- BEST' if bpb == min(v[0] for v in results.values() if v[0]) else ''
    print(f'  {name:20s}  BPB={bpb:.4f}  delta={delta:+.4f}  params={params/1e6:.1f}M  ~{est_mb:.1f}MB  t={t:.0f}s{marker}')
