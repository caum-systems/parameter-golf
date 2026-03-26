"""Quick A/B test: standard CE loss vs byte-weighted CE loss."""
import sys, os, time, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
os.environ['DATA_PATH'] = './data/datasets/fineweb10B_sp1024'

from train_gpt_depthrecur import (
    GPT, Hyperparameters, load_data_shard, load_validation_tokens,
    build_sentencepiece_luts
)
from pathlib import Path
import sentencepiece as spm

args = Hyperparameters()
device = 'cuda'
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, args.vocab_size, device
)

# Show byte distribution
bw = base_bytes_lut.float()
print(f"Byte weights: min={bw.min():.0f} max={bw.max():.0f} mean={bw.mean():.2f}")
print(f"  1-byte tokens: {(bw <= 1).sum().item()}")
print(f"  2-byte tokens: {(bw == 2).sum().item()}")
print(f"  3-byte tokens: {(bw == 3).sum().item()}")
print(f"  4+ byte tokens: {(bw >= 4).sum().item()}")

# Build model
def make_model():
    return GPT(
        vocab_size=1024, num_layers=12, model_dim=768, num_heads=12,
        num_kv_heads=4, mlp_mult=3, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, bigram_vocab_size=2048, bigram_dim=128,
        xsa_last_n=4, rope_dims=16, ln_scale=True,
        ve_enabled=True, ve_dim=128, ve_layers='10,11',
        num_unique_blocks=4, repeats=3, lora_rank=16,
        deep_sup_enabled=False,
    ).to(device).bfloat16()

shard = load_data_shard(Path(args.data_path) / 'fineweb_train_000000.bin')
val_tokens = load_validation_tokens(args.val_files, 512)
seq_len = 512
steps = 500
batch = 8

def train_and_eval(model, use_byte_weights):
    if use_byte_weights:
        bw_norm = base_bytes_lut.float().clamp(min=1.0)
        bw_norm = bw_norm / bw_norm.mean()
        model.register_buffer('_byte_weights_lut', bw_norm, persistent=False)
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
    # Eval: compute BPB on validation
    model.eval()
    total_nll = 0.0
    total_bytes = 0.0
    total_tokens_count = 0
    n_eval = min(50, (val_tokens.numel() - 1) // seq_len)
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
            # BPB calculation
            targets_flat = y_b.reshape(-1)
            byte_counts = base_bytes_lut[targets_flat].float()
            # Add leading space bytes
            prev_flat = x_b.reshape(-1)
            byte_counts += (has_leading_space_lut[targets_flat] & ~is_boundary_token_lut[prev_flat]).float()
            total_nll += nll.sum().item()
            total_bytes += byte_counts.sum().item()
            total_tokens_count += bs * seq_len
    val_loss = total_nll / total_tokens_count
    val_bpb = (total_nll / total_bytes) / math.log(2.0)
    return val_loss, val_bpb, train_time

# Save same init for fair comparison
torch.manual_seed(42)
model_a = make_model()
init_state = {k: v.detach().cpu().clone() for k, v in model_a.state_dict().items()}

# Test A: Standard loss
print(f"\n=== TEST A: Standard CE Loss ({steps} steps) ===")
model_a.load_state_dict({k: v.to(device) for k, v in init_state.items()})
loss_a, bpb_a, time_a = train_and_eval(model_a, use_byte_weights=False)
print(f"  val_loss={loss_a:.4f} val_bpb={bpb_a:.6f} time={time_a:.1f}s")

# Test B: Byte-weighted loss
print(f"\n=== TEST B: Byte-Weighted CE Loss ({steps} steps) ===")
model_b = make_model()
model_b.load_state_dict({k: v.to(device) for k, v in init_state.items()})
loss_b, bpb_b, time_b = train_and_eval(model_b, use_byte_weights=True)
print(f"  val_loss={loss_b:.4f} val_bpb={bpb_b:.6f} time={time_b:.1f}s")

print(f"\n{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
delta = bpb_b - bpb_a
winner = "BYTE-WEIGHTED WINS" if delta < 0 else "STANDARD WINS" if delta > 0 else "TIE"
print(f"  Standard BPB:      {bpb_a:.6f}")
print(f"  Byte-Weighted BPB: {bpb_b:.6f}")
print(f"  Delta BPB:         {delta:+.6f}  --> {winner}")
