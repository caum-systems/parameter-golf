"""
FULL IDEA TOURNAMENT
=====================

Test ALL remaining ideas on the same mini-training benchmark.
Each idea gets 100 steps on synthetic data with a tiny model.
Winner gets included in the submission stack.

Ideas tested:
1. Inception Refinement Head (already won, included as reference)
2. The Prestige (dual expert heads with routing)
3. Self-Distillation (model teaches itself mid-training)
4. Adaptive LR (adjust LR based on loss dynamics)
5. Future Prediction Head (predict tokens 5-10 ahead as auxiliary loss)
6. Progressive Sequence Length (start short, go long)
7. Confidence-Weighted Loss (weight hard tokens more)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import copy

torch.manual_seed(42)

# ============================================================
# SHARED: Tiny model and data
# ============================================================

class MiniAttn(nn.Module):
    def __init__(self, dim, heads=2):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv(x).reshape(B,T,3,self.heads,self.hd)
        q,k,v = qkv.unbind(2)
        q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.proj(y.transpose(1,2).reshape(B,T,C))

class MiniMLP(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.fc = nn.Linear(dim, dim*mult, bias=False)
        self.proj = nn.Linear(dim*mult, dim, bias=False)
    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))

class TinyGPT(nn.Module):
    def __init__(self, dim=128, vocab=512, layers=3, heads=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                'n1': nn.LayerNorm(dim), 'attn': MiniAttn(dim, heads),
                'n2': nn.LayerNorm(dim), 'mlp': MiniMLP(dim),
            }))
        self.norm = nn.LayerNorm(dim)
        self.vocab = vocab
        self.dim = dim
    
    def forward(self, idx, targets=None, return_hidden=False):
        x = self.emb(idx)
        for b in self.blocks:
            x = x + b['attn'](b['n1'](x))
            x = x + b['mlp'](b['n2'](x))
        x = self.norm(x)
        
        if return_hidden:
            return x
        
        logits = F.linear(x, self.emb.weight)
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab), targets.reshape(-1))
            return loss
        return logits

def make_data(vocab=512, n_seq=500, seq_len=48):
    """Synthetic data with mixed patterns."""
    torch.manual_seed(42)
    data = torch.randint(0, vocab, (n_seq, seq_len + 1))
    # Add repetitive patterns (30% of sequences)
    for i in range(0, n_seq, 3):
        plen = torch.randint(3, 10, (1,)).item()
        pat = torch.randint(0, vocab, (plen,))
        for j in range(seq_len + 1):
            data[i, j] = pat[j % plen]
    # Add bigram patterns (20% of sequences)
    for i in range(1, n_seq, 5):
        for j in range(1, seq_len + 1):
            if torch.rand(1) < 0.5:
                data[i, j] = (data[i, j-1] + 7) % vocab
    return data

DIM = 128
VOCAB = 512
LAYERS = 3
STEPS = 150
BS = 16
SEQ = 48
DATA = make_data(VOCAB, 500, SEQ)

def train_baseline():
    """Train baseline model, return losses."""
    torch.manual_seed(42)
    model = TinyGPT(DIM, VOCAB, LAYERS)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses = []
    for step in range(STEPS):
        idx = torch.randint(0, 500, (BS,))
        batch = DATA[idx]
        loss = model(batch[:, :-1], batch[:, 1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses, model

# ============================================================
# IDEA 2: The Prestige (Dual Expert Heads)
# ============================================================

class TinyGPT_DualHead(nn.Module):
    """Two expert output heads with learned routing."""
    def __init__(self, dim=128, vocab=512, layers=3, heads=2, num_experts=2):
        super().__init__()
        self.base = TinyGPT(dim, vocab, layers, heads)
        self.num_experts = num_experts
        
        # Router
        self.router = nn.Linear(dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        
        # Expert-specific scales (each expert sees hidden states differently)
        self.expert_scales = nn.Parameter(
            torch.ones(num_experts, dim) + torch.randn(num_experts, dim) * 0.01
        )
        self.vocab = vocab
    
    def forward(self, idx, targets=None):
        hidden = self.base(idx, return_hidden=True)  # [B, T, D]
        B, T, D = hidden.shape
        
        # Route
        pi = F.softmax(self.router(hidden.mean(dim=1)), dim=-1)  # [B, K]
        
        # Each expert computes logits with different scaling
        log_probs_list = []
        for k in range(self.num_experts):
            h_k = hidden * self.expert_scales[k]
            logits_k = F.linear(h_k, self.base.emb.weight)
            log_probs_list.append(F.log_softmax(logits_k, dim=-1))  # [B, T, V]
        
        # Mix in log space
        all_log_probs = torch.stack(log_probs_list, dim=0)  # [K, B, T, V]
        log_pi = torch.log(pi.T.unsqueeze(-1).unsqueeze(-1) + 1e-8)  # [K, B, 1, 1]
        log_mixture = torch.logsumexp(log_pi + all_log_probs, dim=0)  # [B, T, V]
        
        if targets is not None:
            loss = F.nll_loss(log_mixture.reshape(-1, self.vocab), targets.reshape(-1))
            return loss
        return log_mixture

def test_dual_head():
    torch.manual_seed(42)
    model = TinyGPT_DualHead(DIM, VOCAB, LAYERS)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses = []
    for step in range(STEPS):
        idx = torch.randint(0, 500, (BS,))
        batch = DATA[idx]
        loss = model(batch[:, :-1], batch[:, 1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses, model


# ============================================================
# IDEA 3: Self-Distillation (train on own predictions mid-run)
# ============================================================

def test_self_distillation():
    torch.manual_seed(42)
    model = TinyGPT(DIM, VOCAB, LAYERS)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses = []
    teacher_state = None
    
    for step in range(STEPS):
        idx = torch.randint(0, 500, (BS,))
        batch = DATA[idx]
        x, y = batch[:, :-1], batch[:, 1:]
        
        # Standard CE loss
        logits = model(x)
        ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        
        total_loss = ce_loss
        
        # Every 30 steps: snapshot teacher and add KD loss
        if step > 0 and step % 30 == 0:
            teacher_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if teacher_state is not None and step % 30 > 5:
            # Load teacher weights into a copy
            with torch.no_grad():
                teacher = TinyGPT(DIM, VOCAB, LAYERS)
                teacher.load_state_dict(teacher_state)
                teacher_logits = teacher(x)
                teacher_probs = F.softmax(teacher_logits / 2.0, dim=-1)  # temperature=2
            
            student_log_probs = F.log_softmax(logits / 2.0, dim=-1)
            kd_loss = F.kl_div(
                student_log_probs.reshape(-1, VOCAB),
                teacher_probs.reshape(-1, VOCAB),
                reduction='batchmean'
            ) * 4.0  # scale by T^2
            
            total_loss = 0.9 * ce_loss + 0.1 * kd_loss
        
        opt.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(ce_loss.item())
    
    return losses, model


# ============================================================
# IDEA 4: Adaptive LR based on loss dynamics
# ============================================================

def test_adaptive_lr():
    torch.manual_seed(42)
    model = TinyGPT(DIM, VOCAB, LAYERS)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses = []
    loss_history = []
    base_lr = 3e-3
    
    for step in range(STEPS):
        idx = torch.randint(0, 500, (BS,))
        batch = DATA[idx]
        loss = model(batch[:, :-1], batch[:, 1:])
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        loss_history.append(loss.item())
        
        # Adaptive LR: every 10 steps, check loss dynamics
        if step > 20 and step % 10 == 0:
            recent = sum(loss_history[-10:]) / 10
            older = sum(loss_history[-20:-10]) / 10
            
            if recent > older * 0.999:
                # Loss not decreasing → reduce LR
                for pg in opt.param_groups:
                    pg['lr'] = max(pg['lr'] * 0.8, base_lr * 0.1)
            elif recent < older * 0.95:
                # Loss decreasing fast → can increase LR slightly
                for pg in opt.param_groups:
                    pg['lr'] = min(pg['lr'] * 1.1, base_lr * 2.0)
    
    return losses, model


# ============================================================
# IDEA 5: Future Prediction (auxiliary loss predicting ahead)
# ============================================================

class TinyGPT_FuturePred(nn.Module):
    """Auxiliary heads that predict tokens 3, 5, 10 steps ahead."""
    def __init__(self, dim=128, vocab=512, layers=3, heads=2):
        super().__init__()
        self.base = TinyGPT(dim, vocab, layers, heads)
        self.vocab = vocab
        
        # Small projection heads for future prediction
        self.future_heads = nn.ModuleList([
            nn.Linear(dim, vocab, bias=False) for _ in range(3)
        ])
        self.future_offsets = [3, 5, 10]
        
        for h in self.future_heads:
            nn.init.normal_(h.weight, std=0.01)
    
    def forward(self, idx, targets=None):
        hidden = self.base(idx, return_hidden=True)
        logits = F.linear(hidden, self.base.emb.weight)
        
        if targets is None:
            return logits
        
        # Main loss
        main_loss = F.cross_entropy(logits.reshape(-1, self.vocab), targets.reshape(-1))
        
        # Future prediction losses
        aux_loss = 0.0
        B, T, D = hidden.shape
        
        for head, offset in zip(self.future_heads, self.future_offsets):
            if T > offset:
                # Predict token at position t+offset using hidden state at position t
                future_logits = head(hidden[:, :-offset, :])  # [B, T-offset, V]
                future_targets = targets[:, offset:]           # [B, T-offset]
                
                min_len = min(future_logits.shape[1], future_targets.shape[1])
                fl = future_logits[:, :min_len, :]
                ft = future_targets[:, :min_len]
                
                if min_len > 0:
                    aux_loss += F.cross_entropy(
                        fl.reshape(-1, self.vocab), ft.reshape(-1)
                    ) * 0.05  # small weight
        
        return main_loss + aux_loss

def test_future_pred():
    torch.manual_seed(42)
    model = TinyGPT_FuturePred(DIM, VOCAB, LAYERS)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses = []
    for step in range(STEPS):
        idx = torch.randint(0, 500, (BS,))
        batch = DATA[idx]
        loss = model(batch[:, :-1], batch[:, 1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # Track only main loss for fair comparison
        with torch.no_grad():
            main_loss = F.cross_entropy(
                F.linear(model.base(batch[:, :-1], return_hidden=True), model.base.emb.weight).reshape(-1, VOCAB),
                batch[:, 1:].reshape(-1)
            )
        losses.append(main_loss.item())
    return losses, model


# ============================================================
# IDEA 6: Confidence-Weighted Loss
# ============================================================

def test_confidence_weighted():
    """Weight hard-to-predict tokens more in the loss."""
    torch.manual_seed(42)
    model = TinyGPT(DIM, VOCAB, LAYERS)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses = []
    
    for step in range(STEPS):
        idx = torch.randint(0, 500, (BS,))
        batch = DATA[idx]
        x, y = batch[:, :-1], batch[:, 1:]
        
        logits = model(x)
        
        # Compute per-token loss
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), y.reshape(-1), reduction='none'
        )
        
        # Weight: tokens the model is uncertain about get higher weight
        # This focuses learning on the hard cases
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            max_prob = probs.reshape(-1, VOCAB).max(dim=-1).values
            # High confidence → low weight, low confidence → high weight
            weights = 2.0 - max_prob  # range [1.0, 2.0]
            weights = weights / weights.mean()  # normalize
        
        loss = (per_token_loss * weights).mean()
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        # Track unweighted loss for fair comparison
        losses.append(per_token_loss.mean().item())
    
    return losses, model


# ============================================================
# IDEA 7: Stochastic Depth (randomly skip layers during training)
# ============================================================

class TinyGPT_StochasticDepth(nn.Module):
    def __init__(self, dim=128, vocab=512, layers=3, heads=2, drop_rate=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList()
        self.drop_probs = []
        for i in range(layers):
            self.blocks.append(nn.ModuleDict({
                'n1': nn.LayerNorm(dim), 'attn': MiniAttn(dim, heads),
                'n2': nn.LayerNorm(dim), 'mlp': MiniMLP(dim),
            }))
            # Linear increase in drop probability
            self.drop_probs.append(drop_rate * (i + 1) / layers)
        self.norm = nn.LayerNorm(dim)
        self.vocab = vocab
    
    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b, dp in zip(self.blocks, self.drop_probs):
            if self.training and torch.rand(1).item() < dp:
                continue  # skip this layer
            x = x + b['attn'](b['n1'](x))
            x = x + b['mlp'](b['n2'](x))
        x = self.norm(x)
        logits = F.linear(x, self.emb.weight)
        if targets is not None:
            return F.cross_entropy(logits.reshape(-1, self.vocab), targets.reshape(-1))
        return logits

def test_stochastic_depth():
    torch.manual_seed(42)
    model = TinyGPT_StochasticDepth(DIM, VOCAB, LAYERS, drop_rate=0.15)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses = []
    for step in range(STEPS):
        idx = torch.randint(0, 500, (BS,))
        batch = DATA[idx]
        loss = model(batch[:, :-1], batch[:, 1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # Eval without dropout for fair comparison
        model.eval()
        with torch.no_grad():
            eval_loss = model(batch[:, :-1], batch[:, 1:])
        model.train()
        losses.append(eval_loss.item())
    return losses, model


# ============================================================
# TOURNAMENT
# ============================================================

def run_tournament():
    print("=" * 70)
    print("FULL IDEA TOURNAMENT — 150 steps, same data, same seed")
    print("=" * 70)
    
    ideas = [
        ("Baseline (standard)", train_baseline),
        ("Dual Expert Heads", test_dual_head),
        ("Self-Distillation", test_self_distillation),
        ("Adaptive LR", test_adaptive_lr),
        ("Future Prediction", test_future_pred),
        ("Confidence-Weighted Loss", test_confidence_weighted),
        ("Stochastic Depth", test_stochastic_depth),
    ]
    
    results = {}
    
    for name, fn in ideas:
        print(f"\n  Running: {name}...")
        t0 = time.time()
        losses, model = fn()
        t1 = time.time()
        
        final_10 = sum(losses[-10:]) / 10
        min_loss = min(losses)
        params = sum(p.numel() for p in model.parameters())
        
        results[name] = {
            'final_loss': final_10,
            'min_loss': min_loss,
            'params': params,
            'time': t1 - t0,
            'losses': losses,
        }
        
        print(f"    Final loss: {final_10:.4f} | Min: {min_loss:.4f} | "
              f"Params: {params:,} | Time: {t1-t0:.1f}s")
    
    # Rankings
    print(f"\n\n{'='*70}")
    print("FINAL RANKINGS (sorted by final loss)")
    print(f"{'='*70}\n")
    
    baseline_loss = results["Baseline (standard)"]['final_loss']
    baseline_params = results["Baseline (standard)"]['params']
    
    ranked = sorted(results.items(), key=lambda x: x[1]['final_loss'])
    
    print(f"{'Rank':>4} | {'Idea':<30} | {'Final Loss':>10} | {'vs Base':>8} | {'Params':>10} | {'Extra':>8}")
    print("-" * 85)
    
    for rank, (name, r) in enumerate(ranked, 1):
        delta = r['final_loss'] - baseline_loss
        extra = r['params'] - baseline_params
        marker = " ★" if delta < -0.05 else ""
        print(f"{rank:>4} | {name:<30} | {r['final_loss']:>10.4f} | {delta:>+8.4f} | "
              f"{r['params']:>10,} | {extra:>+8,}{marker}")
    
    # Loss curves comparison
    print(f"\n\nLoss curves (every 25 steps):")
    header = f"{'Step':>6}"
    for name in [n for n, _ in ranked[:5]]:
        short = name[:15]
        header += f" | {short:>15}"
    print(header)
    print("-" * len(header))
    
    for step in range(0, STEPS, 25):
        row = f"{step:>6}"
        for name, _ in ranked[:5]:
            row += f" | {results[name]['losses'][step]:>15.4f}"
        print(row)
    
    # Verdict
    print(f"\n\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    
    winners = [(n, r) for n, r in ranked if r['final_loss'] < baseline_loss - 0.05]
    neutrals = [(n, r) for n, r in ranked if abs(r['final_loss'] - baseline_loss) <= 0.05]
    losers = [(n, r) for n, r in ranked if r['final_loss'] > baseline_loss + 0.05]
    
    print(f"\n  WINNERS (>0.05 improvement):")
    for name, r in winners:
        extra_kb = (r['params'] - baseline_params) * 0.70 / 1024
        print(f"    ★ {name}: {r['final_loss'] - baseline_loss:+.4f} loss, "
              f"+{extra_kb:.1f}KB artifact cost")
    
    print(f"\n  NEUTRAL (±0.05):")
    for name, r in neutrals:
        print(f"    ○ {name}: {r['final_loss'] - baseline_loss:+.4f} loss")
    
    print(f"\n  LOSERS (>0.05 worse):")
    for name, r in losers:
        print(f"    ✗ {name}: {r['final_loss'] - baseline_loss:+.4f} loss")
    
    print(f"""

RECOMMENDATION FOR SUBMISSION STACK:
  Include winners + cheap neutrals. Exclude losers.
  
  All modules should have an env var to enable/disable:
    INCEPTION_ENABLED=1   (from previous test)
    DUAL_HEAD_ENABLED=1
    FUTURE_PRED_ENABLED=1
    etc.
  
  This way we can ablate on H100 and keep only what helps.
""")
    
    return results


if __name__ == "__main__":
    results = run_tournament()
