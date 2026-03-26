"""Test: No recurrence (like leaderboard #1) vs best recurrence configs.

Leaderboard #1 uses 11 unique layers × 512d × NO LoRA = 1.1194 BPB
Our best so far: 6×2 × 896d = -0.127 delta at 2K steps

This test compares:
  1. 12×1 × 512d (no recurrence, no LoRA, like #1 but 12 layers)
  2. 11×1 × 512d (exact #1 config)
  3. F_WIDER: 4×3 × 896d + LoRA-16 (our best "safe" config)
  4. 12×1 × 640d (mid-point: more unique + wider)
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

COMMON = dict(
    tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0,
    rope_base=10000.0, qk_gain_init=1.5, bigram_vocab_size=2048,
    bigram_dim=128, xsa_last_n=4, rope_dims=16, ln_scale=True,
    ve_enabled=True, ve_dim=128,
    deep_sup_weight=0.1, gated_attention=False, mtp_num_heads=0,
    value_residual=True, vocab_size=1024,
)

CONFIGS = {
    'A_CURRENT_4x3_768': dict(
        num_layers=12, model_dim=768, num_heads=12, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=4, repeats=3, lora_rank=16,
        ve_layers='10,11', deep_sup_enabled=True,
    ),
    'I_NOREC_11x1_512': dict(
        # Exact leaderboard #1 shape (11L, 512d, no recurrence, no LoRA)
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=11, repeats=1, lora_rank=0,
        ve_layers='9,10', deep_sup_enabled=False,
    ),
    'J_NOREC_12x1_512': dict(
        # 12 unique layers at 512d
        num_layers=12, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=12, repeats=1, lora_rank=0,
        ve_layers='10,11', deep_sup_enabled=False,
    ),
    'K_NOREC_12x1_640': dict(
        # 12 unique layers at 640d — mid-point
        num_layers=12, model_dim=640, num_heads=8, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=12, repeats=1, lora_rank=0,
        ve_layers='10,11', deep_sup_enabled=False,
    ),
    'F_WIDER_4x3_896': dict(
        # Our best safe config from round 2
        num_layers=12, model_dim=896, num_heads=8, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=4, repeats=3, lora_rank=16,
        ve_layers='10,11', deep_sup_enabled=True,
    ),
}


def make_model(cfg):
    return GPT(**COMMON, **cfg).to(device).bfloat16()


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
    est_mb = n_params * 2.94 / 8 / 1024 / 1024
    print(f'  [{label}] params={n_params:,} val_bpb={val_bpb:.4f} ~{est_mb:.1f}MB time={train_time:.1f}s')
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
print('NO-RECURRENCE vs RECURRENCE (2000 steps)')
print(f'{"="*60}')
baseline_bpb = results.get('A_CURRENT_4x3_768', (None,))[0]
for name in sorted(results, key=lambda n: results[n][0] if results[n][0] else 99):
    if results[name][0] is None:
        print(f'  {name:25s}  FAILED')
        continue
    bpb, params, t = results[name]
    delta = bpb - baseline_bpb if baseline_bpb else 0
    est_mb = params * 2.94 / 8 / 1024 / 1024
    marker = ' <-- BEST' if bpb == min(v[0] for v in results.values() if v[0]) else ''
    print(f'  {name:25s}  BPB={bpb:.4f}  delta={delta:+.4f}  params={params/1e6:.1f}M  ~{est_mb:.1f}MB  t={t:.0f}s{marker}')
