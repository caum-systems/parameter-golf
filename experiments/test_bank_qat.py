"""Test Bank QAT impact: train with fake-quantization on bank weights.

The top entry (#2, 1.1233 BPB) uses Late QAT on ALL weights.
Our code only had QAT on CastedLinear (10% of params).
New Bank QAT applies STE fake-quant to the 90% bank weights too.

Test: 2000 steps, activate QAT at step 1400 (70% = simulates late_qat_threshold=0.15)
"""
import sys, os, time, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
os.environ['DATA_PATH'] = './data/datasets/fineweb10B_sp1024'

from train_gpt_depthrecur import GPT, CastedLinear, Hyperparameters, load_data_shard, load_validation_tokens
from pathlib import Path

args = Hyperparameters()
device = 'cuda'
seq_len = 512
steps = 2000
batch = 8
qat_start = 1400  # 70% of training = simulates late QAT

shard = load_data_shard(Path(args.data_path) / 'fineweb_train_000000.bin')
val_tokens = load_validation_tokens(args.val_files, seq_len)
print(f'Data loaded. {val_tokens.numel():,} val tokens')


def make_model():
    return GPT(
        vocab_size=1024, num_layers=12, model_dim=768, num_heads=12,
        num_kv_heads=4, mlp_mult=3, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, bigram_vocab_size=2048, bigram_dim=128,
        xsa_last_n=4, rope_dims=16, ln_scale=True,
        ve_enabled=True, ve_dim=128, ve_layers='10,11',
        num_unique_blocks=4, repeats=3, lora_rank=16,
        deep_sup_enabled=True, value_residual=True,
    ).to(device).bfloat16()


def eval_bpb(model):
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
    return (total_nll / total_tokens) / math.log(2.0)


def train(model, label, use_bank_qat=False, use_casted_qat=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    GPT._bank_qat_enabled = False
    CastedLinear._qat_enabled = False
    t0 = time.time()
    for step in range(steps):
        # Activate QAT at 70% of training
        if step == qat_start:
            if use_bank_qat:
                GPT._bank_qat_enabled = True
            if use_casted_qat:
                CastedLinear._qat_enabled = True
            if use_bank_qat or use_casted_qat:
                print(f'  [{label}] QAT activated at step {step}')
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
            print(f'  [{label}] step={step+1} loss={loss.item():.4f}')
    train_time = time.time() - t0
    # Reset QAT flags
    GPT._bank_qat_enabled = False
    CastedLinear._qat_enabled = False
    bpb = eval_bpb(model)
    print(f'  [{label}] val_bpb={bpb:.4f} time={train_time:.1f}s')
    return bpb


# Save same init
torch.manual_seed(42)
init_state = {k: v.cpu().clone() for k, v in make_model().state_dict().items()}

configs = [
    ('NO_QAT',          False, False),
    ('CASTED_QAT_ONLY', False, True),   # current behavior (only 10% of params)
    ('BANK_QAT_ONLY',   True,  False),  # new: 90% of params
    ('FULL_QAT',         True,  True),   # both: 100% of params
]

results = {}
for name, bank_qat, casted_qat in configs:
    print(f'\n=== {name} ===')
    model = make_model()
    model.load_state_dict({k: v.to(device) for k, v in init_state.items()}, strict=False)
    bpb = train(model, name, use_bank_qat=bank_qat, use_casted_qat=casted_qat)
    results[name] = bpb
    del model
    torch.cuda.empty_cache()

print(f'\n{"="*60}')
print(f'BANK QAT COMPARISON (2000 steps, QAT at step {qat_start})')
print(f'{"="*60}')
baseline = results['NO_QAT']
for name, bpb in sorted(results.items(), key=lambda x: x[1]):
    delta = bpb - baseline
    marker = ' <-- BEST' if bpb == min(results.values()) else ''
    print(f'  {name:20s}  BPB={bpb:.4f}  delta={delta:+.4f}{marker}')
