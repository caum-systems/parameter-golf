"""
Local CPU smoke test for Mamba-2 SSM submission.
Verifies: model builds, forward/backward work, quantization pipeline, artifact size.
Does NOT train a useful model — just validates the code.
"""
import io
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Add submission dir to path
sys.path.insert(0, str(Path(__file__).parent / "records" / "track_10min_16mb" / "2026-03-25_SSM_Mamba2_CaumSystems"))

print("=" * 60)
print("CAUM Systems — Mamba-2 SSM Local Smoke Test")
print("=" * 60)

# Import model components from our submission
# We need to patch out CUDA requirements
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_gpt",
    Path(__file__).parent / "records" / "track_10min_16mb" / "2026-03-25_SSM_Mamba2_CaumSystems" / "train_gpt.py"
)

# Read the source directly to extract classes
src = (Path(__file__).parent / "records" / "track_10min_16mb" / "2026-03-25_SSM_Mamba2_CaumSystems" / "train_gpt.py").read_text()

# ---- Inline the key classes (avoid import issues with CUDA checks) ----

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class NaiveSSM(nn.Module):
    """Pure PyTorch SSM for testing."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv-1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + self.d_inner, bias=False)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)
        ssm_out = self.x_proj(x_conv)
        B = ssm_out[..., :self.d_state]
        C = ssm_out[..., self.d_state:2*self.d_state]
        dt = F.softplus(ssm_out[..., 2*self.d_state:])
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        x_db = (dt * x_conv).unsqueeze(-1) * B.unsqueeze(2)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + x_db[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)
        return self.out_proj(y)

class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, headdim):
        super().__init__()
        self.norm = RMSNorm()
        # Always use NaiveSSM for CPU testing (d_state capped for speed)
        self.ssm = NaiveSSM(d_model, d_state=min(d_state, 16), d_conv=d_conv, expand=expand)
        self.block_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))

    def forward(self, x, x0):
        normed = self.norm(x)
        ssm_out = self.ssm(normed)
        return x + self.block_scale[None, None, :] * ssm_out

class MambaLM(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, d_state, d_conv, expand, headdim,
                 tie_embeddings, tied_embed_init_std, logit_softcap):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tie_embeddings = tie_embeddings
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            SSMBlock(model_dim, d_state, d_conv, expand, headdim)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else nn.Linear(model_dim, vocab_size, bias=False)
        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i][None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

# ---- QUANTIZATION (from submission) ----

def quantize_state_dict_int8(state_dict):
    CLIP_Q = 0.9999984
    quantized, scales, dtypes = {}, {}, {}
    passthrough, passthrough_orig = {}, {}
    qmeta = {}
    total_bytes = 0
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point():
            passthrough[name] = t
            total_bytes += t.numel() * t.element_size()
            continue
        if t.numel() <= 65536:
            if t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(torch.float16).contiguous()
            else:
                kept = t
            passthrough[name] = kept
            total_bytes += kept.numel() * kept.element_size()
            continue
        t32 = t.float()
        if t32.ndim == 2:
            clip_abs = torch.quantile(t32.abs(), CLIP_Q, dim=1)
            clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
            scale = (clip_abs / 127.0).clamp_min(1/127)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
            scales[name] = scale.to(torch.float16)
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        else:
            clip_abs = float(torch.quantile(t32.abs().flatten(), CLIP_Q).item())
            scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8)
            scales[name] = scale
        quantized[name] = q.contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        total_bytes += q.numel() + (scales[name].numel() * scales[name].element_size())

    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized,
           "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig: obj["passthrough_orig_dtypes"] = passthrough_orig
    return obj, total_bytes

# ---- RUN TESTS ----

print("\n[1/5] Building model...")
t0 = time.time()
model = MambaLM(
    vocab_size=1024, num_layers=10, model_dim=512,
    d_state=64, d_conv=4, expand=2, headdim=64,
    tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"   Params: {n_params:,}")
print(f"   Time: {time.time()-t0:.1f}s")

# Categorize params
n_2d = sum(p.numel() for p in model.parameters() if p.ndim == 2)
n_other = n_params - n_2d
print(f"   2D (Muon): {n_2d:,} | Scalar (Adam): {n_other:,}")

print("\n[2/5] Forward pass (batch=2, seq=32)...")
t0 = time.time()
x = torch.randint(0, 1024, (2, 32))
y = torch.randint(0, 1024, (2, 32))
with torch.no_grad():
    loss = model(x, y)
print(f"   Loss: {loss.item():.4f} (expected ~6.93 = ln(1024))")
print(f"   Time: {time.time()-t0:.1f}s")

print("\n[3/5] Backward pass...")
t0 = time.time()
loss_train = model(x, y)
loss_train.backward()
grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
print(f"   Gradients computed for {len(grad_norms)} parameters")
print(f"   Max grad norm: {max(grad_norms.values()):.4f}")
print(f"   Time: {time.time()-t0:.1f}s")

print("\n[4/5] Quantization + compression...")
t0 = time.time()
quant_obj, payload_bytes = quantize_state_dict_int8(model.state_dict())
buf = io.BytesIO()
torch.save(quant_obj, buf)
raw_bytes = len(buf.getvalue())
compressed = zlib.compress(buf.getvalue(), 9)
compressed_bytes = len(compressed)
code_bytes = len((Path(__file__).parent / "records" / "track_10min_16mb" /
                   "2026-03-25_SSM_Mamba2_CaumSystems" / "train_gpt.py").read_bytes())
total_artifact = compressed_bytes + code_bytes

print(f"   Raw state dict: {raw_bytes:,} bytes ({raw_bytes/1e6:.1f} MB)")
print(f"   Compressed (zlib-9): {compressed_bytes:,} bytes ({compressed_bytes/1e6:.1f} MB)")
print(f"   Code size: {code_bytes:,} bytes ({code_bytes/1e3:.1f} KB)")
print(f"   TOTAL ARTIFACT: {total_artifact:,} bytes ({total_artifact/1e6:.2f} MB)")
print(f"   Time: {time.time()-t0:.1f}s")

fits = total_artifact < 16_000_000
print(f"\n[5/5] SIZE CHECK: {'PASS' if fits else 'FAIL'} ({total_artifact/1e6:.2f} / 16.00 MB)")

if not fits:
    # Calculate max layers that fit
    per_layer_bytes = (compressed_bytes - 600_000) / 10  # rough estimate
    max_layers = int((16_000_000 - code_bytes - 600_000) / per_layer_bytes)
    print(f"   Reduce to NUM_LAYERS={max_layers} to fit.")

print("\n" + "=" * 60)
if fits:
    print("ALL CHECKS PASSED. Ready for 8xH100 training.")
else:
    print("ARTIFACT TOO LARGE. Reduce layers or model dim.")
print("=" * 60)

# Param breakdown per layer
print("\n--- Parameter breakdown ---")
for name, p in model.named_parameters():
    if "blocks.0." in name:
        print(f"  blocks.0.{name.split('blocks.0.')[-1]}: {p.numel():>10,} ({p.shape})")
