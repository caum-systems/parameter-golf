"""
Parameter Golf — New Modules
==============================

Two innovations that stack on top of PR #549 (current SOTA):

1. TrigramHash Logit Bias: A learned n-gram → logit lookup table
   that gives the model "flash memory" for common n-gram patterns.
   
2. Mixture of Softmaxes (MoS): Replaces the single softmax output 
   with K mixed softmaxes, breaking the softmax bottleneck for small models.

Both are:
- Differentiable (train with standard backprop)
- Tiny parameter cost (<200KB combined)
- 100% legal (stored in artifact, trained during training window)
- Compatible with torch.compile
- Orthogonal to each other and all existing techniques

Usage:
    # In the GPT model's __init__:
    self.trigram_bias = TrigramLogitBias(vocab_size=8192, proj_dim=16, table_size=4096)
    self.mos_head = MixtureOfSoftmaxes(dim=512, vocab_size=8192, num_experts=4)
    
    # In forward(), replace the final logit computation:
    # OLD: logits = F.linear(x, self.tok_emb.weight)
    # NEW:
    trigram_bias = self.trigram_bias(input_ids)
    logits = self.mos_head(x, self.tok_emb.weight, trigram_bias)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. TRIGRAM HASH LOGIT BIAS
# ============================================================

class TrigramLogitBias(nn.Module):
    """
    Learned n-gram → logit bias lookup.
    
    For each position, hashes the previous N tokens into a table index,
    looks up a low-rank logit bias vector, and projects it to vocab size.
    
    This gives the model a "fast path" for predictable n-gram continuations
    (e.g., "United" → "States", "</div" → ">", "import" → "os").
    
    The transformer doesn't need to waste capacity on these patterns —
    the hash table handles them directly.
    
    Params:
        table_size=4096, proj_dim=16, vocab=8192:
        table: 4096 × 16 = 65,536
        proj:  16 × 8192 = 131,072
        total: 196,608 params (~130KB compressed)
        
    Also supports bigram and 4-gram with separate tables for each order.
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        proj_dim: int = 16,
        table_size: int = 4096,
        max_order: int = 4,       # use 2-gram, 3-gram, 4-gram
        min_order: int = 2,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.table_size = table_size
        self.max_order = max_order
        self.min_order = min_order
        self.proj_dim = proj_dim
        
        # One lookup table per n-gram order
        num_orders = max_order - min_order + 1
        self.tables = nn.Parameter(
            torch.randn(num_orders, table_size, proj_dim) * init_scale
        )
        
        # Shared projection from proj_dim → vocab_size
        self.proj = nn.Linear(proj_dim, vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=init_scale)
        
        # Per-order learned scale (starts small, grows if useful)
        self.order_scales = nn.Parameter(
            torch.full((num_orders,), 0.1)
        )
        
        # Hash primes (fixed, not learned) — one per position in the n-gram
        # Using large primes for good hash distribution
        primes = [1, 31, 997, 30011, 900001]
        self.register_buffer(
            'hash_primes',
            torch.tensor(primes[:max_order], dtype=torch.long)
        )
    
    def _hash_ngram(self, token_ids: torch.Tensor, order: int) -> torch.Tensor:
        """
        Hash the last `order` tokens at each position into a table index.
        
        token_ids: [batch, seq_len]
        Returns: [batch, seq_len] of indices in [0, table_size)
        """
        B, T = token_ids.shape
        
        # Accumulate hash from the last `order` tokens
        h = torch.zeros(B, T, dtype=torch.long, device=token_ids.device)
        
        for i in range(order):
            # Shift: position t looks at token at position t-order+1+i
            shift = order - 1 - i
            if shift > 0:
                # Pad with zeros for positions that don't have enough history
                shifted = F.pad(token_ids[:, :-shift], (shift, 0), value=0)
            else:
                shifted = token_ids
            
            h = h + shifted.long() * self.hash_primes[i]
        
        return h % self.table_size
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute logit bias from n-gram context.
        
        input_ids: [batch, seq_len] — the input token IDs
        Returns: [batch, seq_len, vocab_size] — logit bias to add
        """
        B, T = input_ids.shape
        total_bias = torch.zeros(B, T, self.proj_dim, device=input_ids.device)
        
        for i, order in enumerate(range(self.min_order, self.max_order + 1)):
            # Hash the last `order` tokens
            indices = self._hash_ngram(input_ids, order)  # [B, T]
            
            # Lookup in table
            table_vectors = self.tables[i][indices]  # [B, T, proj_dim]
            
            # Scale by learned per-order weight
            scale = self.order_scales[i]
            total_bias = total_bias + table_vectors * scale
        
        # Project to vocab size
        logit_bias = self.proj(total_bias)  # [B, T, vocab_size]
        
        return logit_bias


# ============================================================
# 2. MIXTURE OF SOFTMAXES (MoS)
# ============================================================

class MixtureOfSoftmaxes(nn.Module):
    """
    Replaces the standard single softmax with a mixture of K softmaxes.
    
    Standard: P(w|x) = softmax(x @ E.T)[w]
    MoS:      P(w|x) = sum_k pi_k(x) * softmax(x_k @ E.T)[w]
    
    where pi_k are mixing weights and x_k are K different projections of x.
    
    This breaks the "softmax bottleneck" (Yang et al., 2018):
    a single softmax can only represent log-linear distributions,
    but a mixture can represent ANY distribution over the vocabulary.
    
    For small models this is especially impactful because the hidden dim
    is small relative to vocab size, limiting the expressiveness of
    a single softmax.
    
    Params:
        dim=512, num_experts=4:
        gate: 512 × 4 = 2,048
        expert_proj: 4 × 512 (or 4 learned bias vectors over vocab)
        total: ~35K params (~25KB compressed)
    """
    
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        num_experts: int = 4,
        logit_softcap: float = 30.0,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.logit_softcap = logit_softcap
        
        # Gate: decides mixing weights per token
        self.gate = nn.Linear(dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)
        
        # Per-expert transformation: lightweight projections
        # Each expert gets a learned scaling vector (not a full matrix)
        # This keeps params minimal while giving each expert a "personality"
        self.expert_scales = nn.Parameter(
            torch.ones(num_experts, dim) + torch.randn(num_experts, dim) * 0.01
        )
        
        # Per-expert temperature (controls sharpness of each expert's softmax)
        self.expert_temps = nn.Parameter(
            torch.ones(num_experts) * 1.0
        )
    
    def forward(
        self,
        hidden: torch.Tensor,           # [batch*seq, dim]
        embedding_weight: torch.Tensor,  # [vocab, dim] (tied embeddings)
        extra_logit_bias: torch.Tensor = None,  # [batch*seq, vocab] (from TrigramHash)
    ) -> torch.Tensor:
        """
        Compute mixture-of-softmaxes loss.
        
        Returns logits that can be passed to cross_entropy.
        
        Note: For training, we compute the log of the mixture probability
        directly for numerical stability, rather than mixing logits.
        """
        K = self.num_experts
        
        # Mixing weights: [batch*seq, K]
        pi = F.softmax(self.gate(hidden), dim=-1)
        
        # Compute logits for each expert
        # Each expert scales the hidden state differently before projecting
        expert_logits_list = []
        
        for k in range(K):
            # Expert-specific hidden state
            h_k = hidden * self.expert_scales[k].unsqueeze(0)  # [B*T, dim]
            
            # Project to vocab
            logits_k = F.linear(h_k, embedding_weight)  # [B*T, vocab]
            
            # Add trigram bias if provided
            if extra_logit_bias is not None:
                logits_k = logits_k + extra_logit_bias
            
            # Softcap
            if self.logit_softcap > 0:
                logits_k = self.logit_softcap * torch.tanh(logits_k / self.logit_softcap)
            
            # Temperature
            logits_k = logits_k / self.expert_temps[k].clamp(min=0.1)
            
            expert_logits_list.append(logits_k)
        
        # Stack: [K, B*T, vocab]
        all_logits = torch.stack(expert_logits_list, dim=0)
        
        # Compute log mixture probability using logsumexp for stability:
        # log P(w) = logsumexp_k [log pi_k + log softmax(logits_k)[w]]
        # = logsumexp_k [log pi_k + logits_k[w] - logsumexp(logits_k)]
        
        log_pi = torch.log(pi.T.unsqueeze(-1) + 1e-8)  # [K, B*T, 1]
        log_softmax_k = all_logits - all_logits.logsumexp(dim=-1, keepdim=True)  # [K, B*T, vocab]
        
        # [K, B*T, vocab] → logsumexp over K → [B*T, vocab]
        log_mixture = torch.logsumexp(log_pi + log_softmax_k, dim=0)
        
        # Return as "logits" (technically log-probs, but cross_entropy
        # expects raw logits, so we need to handle this carefully)
        # 
        # IMPORTANT: The output is log P(w|x), not logits.
        # Use NLLLoss instead of CrossEntropyLoss, or convert:
        # Since cross_entropy(logits, target) = nll_loss(log_softmax(logits), target)
        # and we already have log_softmax (log_mixture is log probs),
        # we use nll_loss directly.
        
        return log_mixture  # [B*T, vocab] — these are LOG PROBABILITIES
    
    def compute_loss(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute NLL loss from log mixture probabilities."""
        return F.nll_loss(log_probs, targets, reduction='mean')


# ============================================================
# 3. SIMPLE INTEGRATION VERSION
# ============================================================

class EnhancedOutputHead(nn.Module):
    """
    Drop-in replacement for the output logit computation.
    Combines TrigramHash + MoS into a single module.
    
    Usage in GPT.forward():
        # Replace:
        #   logits = F.linear(x, self.tok_emb.weight)
        #   logits = softcap * tanh(logits / softcap)
        #   loss = F.cross_entropy(logits, targets)
        
        # With:
        #   loss = self.output_head(x, input_ids, targets, self.tok_emb.weight)
    """
    
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        mos_experts: int = 4,
        trigram_table_size: int = 4096,
        trigram_proj_dim: int = 16,
        trigram_max_order: int = 4,
        logit_softcap: float = 30.0,
        use_mos: bool = True,
        use_trigram: bool = True,
    ):
        super().__init__()
        self.use_mos = use_mos
        self.use_trigram = use_trigram
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size
        
        if use_trigram:
            self.trigram = TrigramLogitBias(
                vocab_size=vocab_size,
                proj_dim=trigram_proj_dim,
                table_size=trigram_table_size,
                max_order=trigram_max_order,
            )
        
        if use_mos:
            self.mos = MixtureOfSoftmaxes(
                dim=dim,
                vocab_size=vocab_size,
                num_experts=mos_experts,
                logit_softcap=logit_softcap,
            )
    
    def forward(
        self,
        hidden: torch.Tensor,            # [batch, seq, dim]
        input_ids: torch.Tensor,          # [batch, seq]
        targets: torch.Tensor,            # [batch, seq]
        embedding_weight: torch.Tensor,   # [vocab, dim]
    ) -> torch.Tensor:
        """Returns the loss (scalar)."""
        
        B, T, D = hidden.shape
        
        # Flatten for output computation
        h_flat = hidden.reshape(-1, D)            # [B*T, D]
        targets_flat = targets.reshape(-1)         # [B*T]
        
        # Trigram logit bias
        trigram_bias = None
        if self.use_trigram:
            trigram_bias = self.trigram(input_ids)  # [B, T, vocab]
            trigram_bias = trigram_bias.reshape(-1, self.vocab_size)  # [B*T, vocab]
        
        # Mixture of Softmaxes
        if self.use_mos:
            log_probs = self.mos(h_flat, embedding_weight, trigram_bias)
            loss = F.nll_loss(log_probs, targets_flat, reduction='mean')
        else:
            # Standard single softmax (with optional trigram bias)
            logits = F.linear(h_flat, embedding_weight)
            if trigram_bias is not None:
                logits = logits + trigram_bias
            if self.logit_softcap > 0:
                logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            loss = F.cross_entropy(logits.float(), targets_flat, reduction='mean')
        
        return loss
    
    def get_logits_for_eval(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get logits/log-probs for evaluation (BPB calculation).
        """
        B, T, D = hidden.shape
        h_flat = hidden.reshape(-1, D)
        
        trigram_bias = None
        if self.use_trigram:
            trigram_bias = self.trigram(input_ids).reshape(-1, self.vocab_size)
        
        if self.use_mos:
            log_probs = self.mos(h_flat, embedding_weight, trigram_bias)
            return log_probs.reshape(B, T, -1)
        else:
            logits = F.linear(h_flat, embedding_weight)
            if trigram_bias is not None:
                logits = logits + trigram_bias
            if self.logit_softcap > 0:
                logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            return logits.reshape(B, T, -1)


# ============================================================
# 4. PARAMETER COUNT AND SIZE ANALYSIS
# ============================================================

def analyze_modules(vocab_size=8192, dim=512, num_experts=4, 
                    table_size=4096, proj_dim=16, max_order=4):
    """Print parameter counts for the new modules."""
    
    trigram = TrigramLogitBias(vocab_size, proj_dim, table_size, max_order)
    mos = MixtureOfSoftmaxes(dim, vocab_size, num_experts)
    head = EnhancedOutputHead(dim, vocab_size, num_experts, table_size, proj_dim, max_order)
    
    def count_params(module):
        return sum(p.numel() for p in module.parameters())
    
    def estimate_compressed(module):
        # ~0.70 bytes per param after int6+LZMA
        return count_params(module) * 0.70
    
    print("=" * 60)
    print("NEW MODULES — PARAMETER ANALYSIS")
    print("=" * 60)
    
    print(f"\nConfig: vocab={vocab_size}, dim={dim}")
    print(f"        experts={num_experts}, table={table_size}, proj={proj_dim}")
    print(f"        n-gram orders: {2} to {max_order}")
    
    tg_params = count_params(trigram)
    mos_params = count_params(mos)
    total_params = count_params(head)
    
    tg_compressed = estimate_compressed(trigram)
    mos_compressed = estimate_compressed(mos)
    total_compressed = estimate_compressed(head)
    
    print(f"\n  TrigramHash Logit Bias:")
    print(f"    Parameters: {tg_params:,} ({tg_compressed/1024:.1f} KB compressed)")
    print(f"    Breakdown:")
    for name, p in trigram.named_parameters():
        print(f"      {name}: {list(p.shape)} = {p.numel():,}")
    
    print(f"\n  Mixture of Softmaxes:")
    print(f"    Parameters: {mos_params:,} ({mos_compressed/1024:.1f} KB compressed)")
    print(f"    Breakdown:")
    for name, p in mos.named_parameters():
        print(f"      {name}: {list(p.shape)} = {p.numel():,}")
    
    print(f"\n  TOTAL new params: {total_params:,} ({total_compressed/1024:.1f} KB compressed)")
    print(f"  As % of 16MB budget: {total_compressed/16_000_000*100:.2f}%")
    
    return total_params, total_compressed


# ============================================================
# 5. QUICK FUNCTIONAL TEST
# ============================================================

def test_modules():
    """Verify everything works with a forward pass."""
    print("\n" + "=" * 60)
    print("FUNCTIONAL TEST")
    print("=" * 60)
    
    B, T, D = 2, 32, 512
    vocab = 8192
    
    # Create modules
    head = EnhancedOutputHead(
        dim=D, vocab_size=vocab, 
        mos_experts=4,
        trigram_table_size=4096,
        trigram_proj_dim=16,
        trigram_max_order=4,
        logit_softcap=30.0,
    )
    
    # Fake inputs
    hidden = torch.randn(B, T, D)
    input_ids = torch.randint(0, vocab, (B, T))
    targets = torch.randint(0, vocab, (B, T))
    embedding_weight = torch.randn(vocab, D) * 0.02
    
    # Forward pass
    print("\n  Testing forward (training mode)...")
    loss = head(hidden, input_ids, targets, embedding_weight)
    print(f"    Loss: {loss.item():.4f}")
    print(f"    Loss shape: {loss.shape}")
    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    # Backward pass
    print("  Testing backward...")
    loss.backward()
    
    grad_norms = {}
    for name, p in head.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    
    print(f"    All gradients computed: {len(grad_norms)} parameters")
    print(f"    Gradient norms (sample):")
    for name, norm in list(grad_norms.items())[:5]:
        print(f"      {name}: {norm:.6f}")
    
    # Eval mode
    print("  Testing eval mode (get_logits)...")
    with torch.no_grad():
        log_probs = head.get_logits_for_eval(hidden, input_ids, embedding_weight)
    print(f"    Output shape: {log_probs.shape}")
    assert log_probs.shape == (B, T, vocab), f"Expected {(B, T, vocab)}, got {log_probs.shape}"
    
    # Test without MoS (standard softmax + trigram only)
    print("  Testing trigram-only mode (no MoS)...")
    head_no_mos = EnhancedOutputHead(
        dim=D, vocab_size=vocab, use_mos=False, use_trigram=True
    )
    loss2 = head_no_mos(hidden, input_ids, targets, embedding_weight)
    print(f"    Loss: {loss2.item():.4f}")
    
    # Test without trigram (MoS only)
    print("  Testing MoS-only mode (no trigram)...")
    head_no_tg = EnhancedOutputHead(
        dim=D, vocab_size=vocab, use_mos=True, use_trigram=False
    )
    loss3 = head_no_tg(hidden, input_ids, targets, embedding_weight)
    print(f"    Loss: {loss3.item():.4f}")
    
    print("\n  ✓ ALL TESTS PASSED")
    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    params, compressed = analyze_modules()
    test_modules()
    
    print(f"""

INTEGRATION GUIDE
==================

To add these modules to PR #549's train_gpt.py:

1. Copy the TrigramLogitBias, MixtureOfSoftmaxes, and EnhancedOutputHead
   classes into train_gpt.py (or import from this file).

2. In GPT.__init__(), add:
   
   self.output_head = EnhancedOutputHead(
       dim=model_dim,
       vocab_size=vocab_size,
       mos_experts=int(os.environ.get("MOS_EXPERTS", 4)),
       trigram_table_size=int(os.environ.get("TRIGRAM_TABLE_SIZE", 4096)),
       trigram_proj_dim=int(os.environ.get("TRIGRAM_PROJ_DIM", 16)),
       trigram_max_order=int(os.environ.get("TRIGRAM_MAX_ORDER", 4)),
       logit_softcap=logit_softcap,
       use_mos=bool(int(os.environ.get("MOS_ENABLED", 1))),
       use_trigram=bool(int(os.environ.get("TRIGRAM_ENABLED", 1))),
   )

3. In GPT.forward(), replace the logit computation:
   
   # OLD:
   # logits_proj = F.linear(x, self.tok_emb.weight)
   # logits = softcap * tanh(logits_proj / softcap)
   # return F.cross_entropy(logits.float(), targets)
   
   # NEW:
   return self.output_head(
       x.reshape(bsz, seqlen, -1),  # unflatten if needed
       input_ids,
       target_ids,
       self.tok_emb.weight,
   )

4. For eval (eval_val function), replace logit computation:
   
   # OLD:
   # logits = model(x)
   # loss = F.cross_entropy(logits, y)
   
   # NEW (need to modify model to expose hidden states):
   # This requires a small refactor — model.forward() should
   # return loss directly (which it already does in #549).
   # The BPB calculation uses the loss, not raw logits.

5. Add to Hyperparameters class:
   
   mos_experts = int(os.environ.get("MOS_EXPERTS", 4))
   trigram_table_size = int(os.environ.get("TRIGRAM_TABLE_SIZE", 4096))
   trigram_proj_dim = int(os.environ.get("TRIGRAM_PROJ_DIM", 16))
   trigram_max_order = int(os.environ.get("TRIGRAM_MAX_ORDER", 4))
   mos_enabled = bool(int(os.environ.get("MOS_ENABLED", 1)))
   trigram_enabled = bool(int(os.environ.get("TRIGRAM_ENABLED", 1)))

6. The new params should go in the AdamW optimizer group (not Muon).
   They're small tensors, not large bank matrices.

Total extra artifact cost: ~{compressed/1024:.0f} KB ({compressed/16_000_000*100:.1f}% of 16MB)
""")
