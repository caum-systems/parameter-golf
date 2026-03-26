"""Architecture scaling v2: Fix failed configs + explore 6-block winner variants.

Round 1 results: 6 blocks × 2 reps beat 4 × 3 by -0.042 BPB.
Now test: wider 6-block, fixed 896d, and combinations.
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

# Common args shared by all configs
COMMON = dict(
    tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0,
    rope_base=10000.0, qk_gain_init=1.5, bigram_vocab_size=2048,
    bigram_dim=128, xsa_last_n=4, rope_dims=16, ln_scale=True,
    ve_enabled=True, ve_dim=128, ve_layers='10,11',
    deep_sup_weight=0.1, gated_attention=False, mtp_num_heads=0,
    value_residual=True, deep_sup_enabled=True, vocab_size=1024,
)

CONFIGS = {
    # Baseline for reference
    'A_CURRENT_4x3_768': dict(
        num_layers=12, model_dim=768, num_heads=12, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=4, repeats=3, lora_rank=16,
    ),
    # Round 1 winner
    'C_6BLOCKS_6x2_768': dict(
        num_layers=12, model_dim=768, num_heads=12, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=6, repeats=2, lora_rank=16,
    ),
    # Fixed 896d: 8 heads, 4 kv heads, head_dim=112
    'F_WIDER_4x3_896': dict(
        num_layers=12, model_dim=896, num_heads=8, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=4, repeats=3, lora_rank=16,
    ),
    # 6 blocks + wider: the real test
    'G_6BLOCKS_896': dict(
        num_layers=12, model_dim=896, num_heads=8, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=6, repeats=2, lora_rank=16,
    ),
    # 6 blocks + higher LoRA rank
    'H_6BLOCKS_RANK32': dict(
        num_layers=12, model_dim=768, num_heads=12, num_kv_heads=4,
        mlp_mult=3, num_unique_blocks=6, repeats=2, lora_rank=32,
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
print('ARCHITECTURE SCALING v2 (2000 steps, lower BPB = better)')
print(f'{"="*60}')
baseline_bpb = results.get('A_CURRENT_4x3_768', (None,))[0]
for name in CONFIGS:
    if results[name][0] is None:
        print(f'  {name:25s}  FAILED')
        continue
    bpb, params, t = results[name]
    delta = bpb - baseline_bpb if baseline_bpb else 0
    # Better estimate: current 23.1M -> 8.48MB = ~2.94 bits/param compressed
    est_mb = params * 2.94 / 8 / 1024 / 1024
    marker = ' <-- BEST' if bpb == min(v[0] for v in results.values() if v[0]) else ''
    print(f'  {name:25s}  BPB={bpb:.4f}  delta={delta:+.4f}  params={params/1e6:.1f}M  ~{est_mb:.1f}MB  t={t:.0f}s{marker}')
