"""Test hidden features: value_residual, MTP, gated_attention."""
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
steps = 500
batch = 8

shard = load_data_shard(Path(args.data_path) / 'fineweb_train_000000.bin')
val_tokens = load_validation_tokens(args.val_files, seq_len)
print(f'Data loaded. {val_tokens.numel():,} val tokens')


def make_model(value_residual=False, mtp_num_heads=0, gated_attention=False, lora_rank=8):
    return GPT(
        vocab_size=1024, num_layers=12, model_dim=768, num_heads=12,
        num_kv_heads=4, mlp_mult=3, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, bigram_vocab_size=2048, bigram_dim=128,
        xsa_last_n=4, rope_dims=16, ln_scale=True,
        ve_enabled=True, ve_dim=128, ve_layers='10,11',
        num_unique_blocks=4, repeats=3, lora_rank=lora_rank,
        deep_sup_enabled=False,
        value_residual=value_residual,
        mtp_num_heads=mtp_num_heads,
        gated_attention=gated_attention,
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
    train_time = time.time() - t0
    train_loss = loss.item()

    # Quick eval
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
    print(f'  [{label}] params={n_params:,} train_loss={train_loss:.4f} '
          f'val_bpb={val_bpb:.4f} time={train_time:.1f}s')
    return val_bpb


# Deterministic seed
torch.manual_seed(42)
init_state = {k: v.cpu().clone() for k, v in make_model().state_dict().items()}

configs = [
    ('BASELINE',        dict(value_residual=False, mtp_num_heads=0, gated_attention=False)),
    ('VALUE_RESIDUAL',  dict(value_residual=True,  mtp_num_heads=0, gated_attention=False)),
    ('MTP_2',           dict(value_residual=False, mtp_num_heads=2, gated_attention=False)),
    ('GATED_ATTN',      dict(value_residual=False, mtp_num_heads=0, gated_attention=True)),
    ('VR+MTP',          dict(value_residual=True,  mtp_num_heads=2, gated_attention=False)),
]

results = {}
for name, kwargs in configs:
    print(f'\n=== {name} ===')
    model = make_model(**kwargs)
    # Load matching params from init_state (skip new params that don't exist in baseline)
    missing, unexpected = model.load_state_dict(
        {k: v.to(device) for k, v in init_state.items()}, strict=False
    )
    if missing:
        print(f'  New params (randomly init): {len(missing)}')
    bpb = train_and_eval(model, name)
    results[name] = bpb
    del model
    torch.cuda.empty_cache()

print(f'\n{"="*60}')
print('COMPARISON (500 steps, lower BPB = better)')
print(f'{"="*60}')
baseline_bpb = results['BASELINE']
for name, bpb in sorted(results.items(), key=lambda x: x[1]):
    delta = bpb - baseline_bpb
    marker = ' <-- BEST' if bpb == min(results.values()) else ''
    print(f'  {name:20s}  BPB={bpb:.4f}  delta={delta:+.4f}{marker}')
