"""
Ablation sweep: test SOTA techniques one-by-one on RTX 4070 SUPER.
Each config runs 500 steps on random data. We compare relative train loss
to identify which techniques help most.

Uses PyTorch native attention (no flash_attn_3 needed).
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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print("WARNING: Running on CPU — experiments will be slow")

DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" or hasattr(torch, "bfloat16")) else torch.float32

# ---- Shared hyperparams ----
VOCAB = 1024
SEQ_LEN = 256  # shorter for fast local testing
BATCH = 8
STEPS = 500

# ---- Modules from SOTA ----

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

    def forward(self, seq_len: int, device, dtype):
        key = (seq_len, device)
        if key not in self._cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache[key] = (freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :])
        cos, sin = self._cache[key]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
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
    """Learned gate mixing current embedding with previous token's embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""
    def __init__(self, bigram_vocab: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab = bigram_vocab
        self.embed = nn.Embedding(bigram_vocab, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers."""
    def __init__(self, vocab_size: int, ve_dim: int, kv_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, kv_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.proj(self.embed(token_ids))
        return h * self.scale.to(dtype=h.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 qk_gain_init: float = 1.5, rope_dims: int = 0, use_xsa: bool = False):
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
        self.use_xsa = use_xsa

    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.num_heads, self.head_dim)
            v_exp = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.num_heads, self.head_dim)
        else:
            v_exp = v

        # SDPA (uses native PyTorch, works on CPU and CUDA without flash_attn)
        q_t = q.transpose(1, 2)  # (B, H, T, D)
        k_t = k.transpose(1, 2)
        v_t = v_exp.transpose(1, 2)
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        y = y.transpose(1, 2)  # (B, T, H, D)

        if self.use_xsa:
            # XSA: subtract self-value projection
            Hkv = v.size(2)
            group = self.num_heads // Hkv
            y_g = y.reshape(B, T, Hkv, group, self.head_dim)
            vn = F.normalize(v, dim=-1).unsqueeze(-2)
            proj_val = (y_g * vn).sum(dim=-1, keepdim=True) * vn
            y = (y_g - proj_val).reshape(B, T, self.num_heads, self.head_dim)

        y = y.reshape(B, T, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mult: float, activation: str = "relu_sq"):
        super().__init__()
        hidden = int(mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc(x)
        if self.activation == "relu_sq":
            h = torch.relu(h).square()
        elif self.activation == "leaky_relu_sq":
            h = F.leaky_relu(h, negative_slope=0.5).square()
        elif self.activation == "swiglu":
            h1, h2 = h.chunk(2, dim=-1)
            h = F.silu(h1) * h2
        else:
            h = F.silu(h)
        return self.proj(h)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 activation: str = "relu_sq", layer_idx: int = 0,
                 ln_scale: bool = False, rope_dims: int = 0, use_xsa: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads,
                                        rope_dims=rope_dims, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult, activation=activation)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out


class GPTModel(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int = 8, num_kv_heads: int = 4, mlp_mult: float = 3.0,
                 activation: str = "relu_sq", use_smear: bool = False,
                 use_bigram: bool = False, bigram_vocab: int = 2048, bigram_dim: int = 128,
                 use_ve: bool = False, ve_dim: int = 128, ve_layers: str = "",
                 rope_dims: int = 0, ln_scale: bool = False, xsa_last_n: int = 0,
                 logit_softcap: float = 30.0):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)

        self.smear = SmearGate(model_dim) if use_smear else None
        self.bigram = BigramHashEmbedding(bigram_vocab, bigram_dim, model_dim) if use_bigram else None

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(
            torch.ones(min(self.num_encoder_layers, self.num_decoder_layers), model_dim, dtype=torch.float32))

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                  activation=activation, layer_idx=i, ln_scale=ln_scale,
                  rope_dims=rope_dims,
                  use_xsa=(xsa_last_n > 0 and i >= num_layers - xsa_last_n))
            for i in range(num_layers)
        ])

        # Value Embedding
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if use_ve and ve_layers else []
        kv_dim = num_kv_heads * (model_dim // num_heads)
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_scales = nn.ParameterList([
                nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_scales = nn.ParameterList()

        self.final_norm = RMSNorm()

        # Orthogonal init for attention/MLP weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)
                elif m.weight.ndim == 2 and min(m.weight.shape) >= 64:
                    nn.init.orthogonal_(m.weight, gain=1.0)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x

        ve_cache = {}
        skips = []
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def _get_ve(self, layer_idx, input_ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if "ve" not in ve_cache:
            ve_cache["ve"] = self.ve_shared(input_ids)
        idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache["ve"] * self.ve_scales[idx].to(dtype=ve_cache["ve"].dtype)


# ---- Muon optimizer (simplified for single GPU) ----

def zeropower_ns5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
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


# ---- Data loading (from baseline) ----

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
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

    def take(self, n: int) -> Tensor:
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


def get_data_loader(repo_root: Path):
    """Try to load real FineWeb data. Returns (loader_fn, use_real_data)."""
    train_pattern = str(repo_root / "data" / "datasets" / "fineweb10B_sp1024" / "fineweb_train_*.bin")
    train_files = sorted(glob.glob(train_pattern))
    if not train_files:
        return None, False
    stream = TokenStream(train_pattern)
    print(f"  Real data: {len(train_files)} train shards")
    return stream, True


# ---- Experiment configs ----

CONFIGS = {
    "baseline_11L_3x": dict(
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.0, activation="relu_sq",
    ),
    "+leaky_relu_sq": dict(
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.0, activation="leaky_relu_sq",
    ),
    "+smeargate": dict(
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.0, activation="relu_sq", use_smear=True,
    ),
    "+bigram_hash": dict(
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.0, activation="relu_sq", use_bigram=True,
    ),
    "+rope16": dict(
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.0, activation="relu_sq", rope_dims=16,
    ),
    "+ln_scale": dict(
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.0, activation="relu_sq", ln_scale=True,
    ),
    "+xsa4": dict(
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.0, activation="relu_sq", xsa_last_n=4,
    ),
    "+ve_9_10": dict(
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.0, activation="relu_sq", use_ve=True, ve_layers="9,10",
    ),
    "FULL_SOTA": dict(
        num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.0, activation="leaky_relu_sq",
        use_smear=True, use_bigram=True,
        rope_dims=16, ln_scale=True, xsa_last_n=4,
        use_ve=True, ve_layers="9,10",
    ),
}


def run_experiment(name: str, cfg: dict, steps: int = STEPS) -> dict:
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = GPTModel(vocab_size=VOCAB, **cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Optimizer: Muon for 2D matrix, Adam for rest
    # Use id() to avoid tensor comparison issues
    embed_ids = {id(model.tok_emb.weight)}
    embed_params = [model.tok_emb.weight]
    control_keywords = ("scale", "mix", "gain", "gate", "skip", "smear", "bias")
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

    # Try to load real data
    repo_root = Path(__file__).parent.parent
    stream, use_real = get_data_loader(repo_root)

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

        # Muon-style update for matrix params
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
    # Compute artifact size estimate
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
    raw = buf.getvalue()
    comp = zlib.compress(raw, 9)
    artifact_mb = len(comp) / 1e6

    result = {
        "name": name,
        "params": n_params,
        "loss_first5": sum(losses[:5]) / 5,
        "loss_last5": sum(losses[-5:]) / 5,
        "loss_drop": sum(losses[:5]) / 5 - sum(losses[-5:]) / 5,
        "min_loss": min(losses),
        "artifact_mb": round(artifact_mb, 2),
        "ms_per_step": round(elapsed / steps * 1000, 1),
        "total_time_s": round(elapsed, 1),
    }
    print(f"  Final: loss={result['loss_last5']:.4f} | drop={result['loss_drop']:+.4f} | "
          f"artifact={artifact_mb:.1f}MB | {result['ms_per_step']}ms/step")
    return result


def main():
    print("=" * 60)
    print("Parameter Golf — Ablation Sweep")
    print(f"Device: {DEVICE} | Steps: {STEPS} | Batch: {BATCH} | Seq: {SEQ_LEN}")
    print("=" * 60)

    results = {}
    for name, cfg in CONFIGS.items():
        try:
            results[name] = run_experiment(name, cfg)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"name": name, "error": str(e)}
        # Free GPU memory between runs
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<25} {'Params':>10} {'Loss(last5)':>12} {'Drop':>8} "
          f"{'Artifact':>10} {'ms/step':>8}")
    print("-" * 90)
    baseline_loss = results.get("baseline_11L_3x", {}).get("loss_last5", 0)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<25} {'ERROR':>10}")
            continue
        delta = ""
        if baseline_loss and name != "baseline_11L_3x":
            d = r["loss_last5"] - baseline_loss
            delta = f" ({d:+.4f})"
        print(f"{name:<25} {r['params']:>10,} {r['loss_last5']:>12.4f}{delta:>10} "
              f"{r['artifact_mb']:>8.1f}MB {r['ms_per_step']:>7.1f}")
    print("=" * 90)

    # Save results
    out_dir = Path(__file__).parent
    out_file = out_dir / "ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
