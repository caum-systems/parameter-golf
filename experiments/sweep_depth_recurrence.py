"""
Depth Recurrence + FULL_SOTA sweep.
Key idea: run the same block N times instead of having N unique blocks.
This slashes parameter count, allowing wider models in 16MB.

Configs tested:
1. depth_rec_4x3  — 4 unique blocks, each run 3x = 12 effective layers (10.2M params)
2. depth_rec_4x3_wide — same but 768 dim (uses freed param budget)
3. depth_rec_6x2  — 6 unique blocks, each run 2x = 12 effective layers
4. depth_rec_4x3_sota — depth rec + LeakyReLU² + BigramHash + SmearGate
5. depth_rec_4x3_wide_sota — wide + all SOTA techniques
"""
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---- Device setup ----
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("WARNING: Running on CPU")

VOCAB = 1024
SEQ_LEN = 256
BATCH = 8
STEPS = 500

# ---- Reuse modules from ablation_sweep ----

class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim: int, rope_dims: int = 0, base: float = 10000.0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = {}

    def forward(self, seq_len, device, dtype):
        key = (seq_len, device)
        if key not in self._cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache[key] = (freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :])
        cos, sin = self._cache[key]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab = bigram_vocab
        self.embed = nn.Embedding(bigram_vocab, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, qk_gain_init=1.5, rope_dims=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = rope_dims
        self.rotary = Rotary(self.head_dim, rope_dims=rope_dims)

    def forward(self, x):
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.num_heads, self.head_dim)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mult, activation="leaky_relu_sq"):
        super().__init__()
        hidden = int(mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.activation = activation

    def forward(self, x):
        h = self.fc(x)
        if self.activation == "leaky_relu_sq":
            h = F.leaky_relu(h, negative_slope=0.5).square()
        else:
            h = torch.relu(h).square()
        return self.proj(h)


class Block(nn.Module):
    """Single transformer block with learned residual mixing."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult,
                 activation="leaky_relu_sq", rope_dims=0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult, activation=activation)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in))
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out))
        return x_out


class DepthRecurrentGPT(nn.Module):
    """GPT with depth recurrence: N unique blocks run R times each = N*R effective layers.
    U-Net skip connections operate on the effective layer sequence."""

    def __init__(self, vocab_size, num_unique_blocks, repeats, model_dim,
                 num_heads=8, num_kv_heads=4, mlp_mult=3.0,
                 activation="leaky_relu_sq", rope_dims=16,
                 use_smear=False, use_bigram=False,
                 bigram_vocab=2048, bigram_dim=128,
                 logit_softcap=30.0):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_unique_blocks = num_unique_blocks
        self.repeats = repeats
        self.effective_layers = num_unique_blocks * repeats

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)

        self.smear = SmearGate(model_dim) if use_smear else None
        self.bigram = BigramHashEmbedding(bigram_vocab, bigram_dim, model_dim) if use_bigram else None

        # U-Net skip connections over effective layers
        self.num_encoder_layers = self.effective_layers // 2
        self.num_decoder_layers = self.effective_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(
            torch.ones(min(self.num_encoder_layers, self.num_decoder_layers), model_dim, dtype=torch.float32))

        # Only N unique blocks (the key savings)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                  activation=activation, rope_dims=rope_dims)
            for _ in range(num_unique_blocks)
        ])

        # Per-repetition learned scales (tiny cost, helps differentiate passes)
        self.rep_scales = nn.Parameter(torch.ones(self.effective_layers, model_dim, dtype=torch.float32))

        self.final_norm = RMSNorm()

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)
                elif m.weight.ndim == 2 and min(m.weight.shape) >= 64:
                    nn.init.orthogonal_(m.weight, gain=1.0)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x

        # Build effective layer sequence: block0, block1, ..., blockN-1, block0, block1, ...
        skips = []
        eff_idx = 0
        for rep in range(self.repeats):
            for blk_idx in range(self.num_unique_blocks):
                # U-Net encoder phase
                if eff_idx < self.num_encoder_layers:
                    x = self.blocks[blk_idx](x, x0)
                    x = x * self.rep_scales[eff_idx].to(dtype=x.dtype)[None, None, :]
                    skips.append(x)
                else:
                    # U-Net decoder phase
                    dec_idx = eff_idx - self.num_encoder_layers
                    if skips:
                        x = x + self.skip_weights[dec_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                    x = self.blocks[blk_idx](x, x0)
                    x = x * self.rep_scales[eff_idx].to(dtype=x.dtype)[None, None, :]
                eff_idx += 1

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ---- Data loading ----

def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def zeropower_ns5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X /= X.norm() + eps
    tr = G.size(0) > G.size(1)
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if tr else X


# ---- Configs ----

CONFIGS = {
    "rec_4x3_512_sota": dict(
        # Previous winner — baseline for comparison
        num_unique_blocks=4, repeats=3, model_dim=512,
        num_heads=8, num_kv_heads=4, mlp_mult=3.0,
        activation="leaky_relu_sq",
        use_smear=True, use_bigram=True,
    ),
    "rec_4x3_640_sota": dict(
        # HYBRID: my 4x3+SOTA + other agent's 640d sweet spot
        num_unique_blocks=4, repeats=3, model_dim=640,
        num_heads=10, num_kv_heads=5, mlp_mult=3.0,
        activation="leaky_relu_sq",
        use_smear=True, use_bigram=True,
    ),
    "rec_6x2_640_sota": dict(
        # Other agent's best structure + my SOTA techniques
        num_unique_blocks=6, repeats=2, model_dim=640,
        num_heads=10, num_kv_heads=5, mlp_mult=3.0,
        activation="leaky_relu_sq",
        use_smear=True, use_bigram=True,
    ),
    "rec_4x3_704_sota": dict(
        # Push wider — 704 = 64*11, good for head_dim=64
        num_unique_blocks=4, repeats=3, model_dim=704,
        num_heads=11, num_kv_heads=4, mlp_mult=3.0,
        activation="leaky_relu_sq",
        use_smear=True, use_bigram=True,
    ),
    "rec_5x3_640_sota": dict(
        # More unique blocks (5) × 3 = 15 effective layers
        num_unique_blocks=5, repeats=3, model_dim=640,
        num_heads=10, num_kv_heads=5, mlp_mult=3.0,
        activation="leaky_relu_sq",
        use_smear=True, use_bigram=True,
    ),
}


def run_experiment(name, cfg, steps=STEPS):
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = DepthRecurrentGPT(vocab_size=VOCAB, **cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    eff_layers = cfg["num_unique_blocks"] * cfg["repeats"]
    print(f"  Params: {n_params:,} | Unique blocks: {cfg['num_unique_blocks']} | "
          f"Repeats: {cfg['repeats']} | Effective layers: {eff_layers} | Dim: {cfg['model_dim']}")

    # Optimizer split
    embed_ids = {id(model.tok_emb.weight)}
    embed_params = [model.tok_emb.weight]
    control_keywords = ("scale", "mix", "gain", "gate", "skip", "smear", "bias", "rep_scale")
    matrix_params, scalar_params = [], []
    for n, p in model.named_parameters():
        if id(p) in embed_ids:
            continue
        is_control = any(k in n for k in control_keywords)
        if p.ndim == 2 and not is_control and p.numel() > 65536:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    opt_embed = torch.optim.Adam(embed_params, lr=0.05, betas=(0.9, 0.95))
    opt_scalar = torch.optim.Adam(scalar_params, lr=0.025, betas=(0.9, 0.95)) if scalar_params else None
    opt_matrix = torch.optim.SGD(matrix_params, lr=0.01, momentum=0.95)

    # Load real data
    repo_root = Path(__file__).parent.parent
    train_pattern = str(repo_root / "data" / "datasets" / "fineweb10B_sp1024" / "fineweb_train_*.bin")
    train_files = sorted(glob.glob(train_pattern))
    use_real = bool(train_files)
    stream = TokenStream(train_pattern) if use_real else None
    if use_real:
        print(f"  Real data: {len(train_files)} shards")

    model.train()
    losses = []
    t0 = time.time()

    for step in range(steps):
        if use_real:
            chunk = stream.take(BATCH * SEQ_LEN + 1).to(dtype=torch.int64)
            x = chunk[:-1].reshape(BATCH, SEQ_LEN).to(DEVICE)
            y = chunk[1:].reshape(BATCH, SEQ_LEN).to(DEVICE)
        else:
            x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=DEVICE)
            y = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=DEVICE)

        opt_embed.zero_grad()
        if opt_scalar:
            opt_scalar.zero_grad()
        opt_matrix.zero_grad()

        if DEVICE.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
        else:
            loss = model(x, y)

        loss.backward()

        with torch.no_grad():
            for p in matrix_params:
                if p.grad is not None:
                    g = zeropower_ns5(p.grad, steps=5)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    p.grad.copy_(g)

        opt_embed.step()
        if opt_scalar:
            opt_scalar.step()
        opt_matrix.step()

        losses.append(loss.item())
        if step < 3 or step == steps - 1 or (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {step+1:>4}/{steps} | loss: {loss.item():.4f} | "
                  f"{elapsed/(step+1)*1000:.0f}ms/step")

    elapsed = time.time() - t0

    # Artifact size estimate
    import io, zlib
    sd = model.state_dict()
    q_sd = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        if t.is_floating_point() and t.numel() > 100:
            scale = t.abs().max() / 127
            q_sd[k] = (torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8), scale)
        else:
            q_sd[k] = t
    buf = io.BytesIO()
    torch.save(q_sd, buf)
    comp = zlib.compress(buf.getvalue(), 9)
    artifact_mb = len(comp) / 1e6

    result = {
        "name": name,
        "params": n_params,
        "unique_blocks": cfg["num_unique_blocks"],
        "repeats": cfg["repeats"],
        "effective_layers": eff_layers,
        "model_dim": cfg["model_dim"],
        "loss_first5": sum(losses[:5]) / 5,
        "loss_last5": sum(losses[-5:]) / 5,
        "loss_drop": sum(losses[:5]) / 5 - sum(losses[-5:]) / 5,
        "min_loss": min(losses),
        "artifact_mb": round(artifact_mb, 2),
        "ms_per_step": round(elapsed / steps * 1000, 1),
    }
    print(f"  Final: loss={result['loss_last5']:.4f} | drop={result['loss_drop']:+.4f} | "
          f"artifact={artifact_mb:.1f}MB | {result['ms_per_step']}ms/step")
    return result


def main():
    print("=" * 60)
    print("Parameter Golf — Depth Recurrence Sweep")
    print(f"Device: {DEVICE} | Steps: {STEPS} | Batch: {BATCH} | Seq: {SEQ_LEN}")
    print("=" * 60)

    results = {}
    for name, cfg in CONFIGS.items():
        try:
            results[name] = run_experiment(name, cfg)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"name": name, "error": str(e)}
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 100)
    print(f"{'Config':<25} {'Params':>10} {'Eff.Layers':>11} {'Dim':>5} {'Loss':>8} "
          f"{'vs base':>10} {'Artifact':>10} {'ms/step':>8}")
    print("-" * 100)
    base_loss = results.get("baseline_11L_3x", {}).get("loss_last5", 0)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<25} {'ERROR':>10} — {r.get('error', '')[:40]}")
            continue
        delta = ""
        if base_loss and name != "baseline_11L_3x":
            d = r["loss_last5"] - base_loss
            delta = f"({d:+.4f})"
        print(f"{name:<25} {r['params']:>10,} {r['effective_layers']:>11} {r['model_dim']:>5} "
              f"{r['loss_last5']:>8.4f} {delta:>10} {r['artifact_mb']:>8.1f}MB {r['ms_per_step']:>7.1f}")
    print("=" * 100)

    out_file = Path(__file__).parent / "depth_recurrence_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
