"""
SVD FACTORED EMBEDDING
========================

Drop-in replacement for nn.Embedding that uses low-rank factorization:
    E[token] = U[token, :rank] @ diag(S[:rank]) @ V[:rank, :dim]

Standard Embedding (vocab=16384, dim=512):
    Parameters: 16384 × 512 = 8,388,608
    Compressed: ~5.7MB

SVD Factored (vocab=16384, dim=512, rank=64):
    Parameters: 16384×64 + 64 + 64×512 = 1,081,409
    Compressed: ~0.74MB
    
    Savings: 87% fewer params, ~5MB freed in artifact

The factored embedding is trained end-to-end with backprop.
At init, U and V are random (no pre-trained SVD needed).
The model learns the optimal factorization during training.

For tied embeddings (output projection reuses embedding weights),
we reconstruct the full [vocab, dim] matrix on-the-fly for the
logit computation. This costs one matmul but saves massive storage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SVDEmbedding(nn.Module):
    """
    Low-rank factored embedding table.
    
    Instead of storing a full [vocab_size, dim] matrix,
    stores U[vocab_size, rank], S[rank], V[rank, dim].
    
    The embedding for token i is: U[i] * S @ V = (U[i] * S) @ V
    
    For tied embeddings (logit projection), reconstructs the full
    matrix as needed: full_weight = (U * S) @ V → [vocab, dim]
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        rank: int = 64,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.rank = rank
        
        # U: per-token factors [vocab_size, rank]
        self.U = nn.Parameter(torch.randn(vocab_size, rank) * init_std)
        
        # S: singular values [rank] — initialized to 1.0 (uniform scale)
        self.S = nn.Parameter(torch.ones(rank))
        
        # V: shared projection [rank, dim]
        self.V = nn.Parameter(torch.randn(rank, dim) * init_std)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for input token IDs.
        
        input_ids: [batch, seq_len] or any shape of long tensor
        Returns: [*input_ids.shape, dim]
        """
        # U[input_ids]: [*, rank]
        u = self.U[input_ids]
        
        # Scale by singular values: [*, rank] * [rank] = [*, rank]
        us = u * self.S.unsqueeze(0)
        
        # Project to full dim: [*, rank] @ [rank, dim] = [*, dim]
        return us @ self.V
    
    @property
    def weight(self) -> torch.Tensor:
        """
        Reconstruct the full embedding matrix for tied embedding logit projection.
        Returns: [vocab_size, dim]
        
        This is called during the logit computation:
            logits = F.linear(hidden, embedding.weight)
        
        The reconstruction is O(vocab × rank × dim) which for
        vocab=16384, rank=64, dim=512 is ~537M FLOPs.
        That's tiny compared to the transformer forward pass.
        """
        # [vocab, rank] * [rank] → [vocab, rank]
        US = self.U * self.S.unsqueeze(0)
        # [vocab, rank] @ [rank, dim] → [vocab, dim]
        return US @ self.V
    
    def extra_repr(self) -> str:
        return (f"vocab_size={self.vocab_size}, dim={self.dim}, "
                f"rank={self.rank}, "
                f"params={self.vocab_size * self.rank + self.rank + self.rank * self.dim:,} "
                f"(vs full: {self.vocab_size * self.dim:,})")


class SVDEmbeddingWithHash(nn.Module):
    """
    SVD Embedding + BigramHash integration.
    
    Combines the factored embedding with a bigram hash table
    that adds n-gram context to each token's representation.
    This is what PR #549 uses (BigramHashEmbedding) but adapted
    for the SVD factored embedding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        rank: int = 64,
        bigram_vocab: int = 2048,
        bigram_init_scale: float = 0.05,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.svd_emb = SVDEmbedding(vocab_size, dim, rank, init_std)
        
        # BigramHash (same as PR #549)
        self.bigram_vocab = bigram_vocab
        if bigram_vocab > 0:
            self.bigram_emb = nn.Embedding(bigram_vocab, dim)
            nn.init.normal_(self.bigram_emb.weight, std=init_std)
            self.bigram_scale = nn.Parameter(torch.tensor(bigram_init_scale))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings with optional bigram hash context."""
        x = self.svd_emb(input_ids)
        
        if self.bigram_vocab > 0 and input_ids.shape[-1] > 1:
            # Hash adjacent token pairs
            prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)
            bigram_keys = (prev_ids * 31 + input_ids) % self.bigram_vocab
            bigram_embs = self.bigram_emb(bigram_keys)
            x = x + self.bigram_scale * bigram_embs
        
        return x
    
    @property
    def weight(self):
        """For tied embeddings."""
        return self.svd_emb.weight


# ============================================================
# ANALYSIS AND TESTING
# ============================================================

def analyze_savings():
    """Show parameter savings for different configs."""
    print("=" * 70)
    print("SVD EMBEDDING — PARAMETER SAVINGS ANALYSIS")
    print("=" * 70)
    
    configs = [
        (1024, 512, "Current: v=1024, d=512"),
        (8192, 512, "v=8192, d=512"),
        (16384, 512, "v=16384, d=512"),
        (32768, 512, "v=32768, d=512"),
    ]
    
    for vocab, dim, label in configs:
        full_params = vocab * dim
        full_compressed = full_params * 0.68  # int6+LZMA estimate
        
        print(f"\n  {label}:")
        print(f"    Full embedding: {full_params:>12,} params ({full_compressed/1e6:.2f}MB)")
        
        for rank in [32, 64, 96, 128]:
            svd_params = vocab * rank + rank + rank * dim
            svd_compressed = svd_params * 0.72  # slightly worse compression for factored
            savings_pct = (1 - svd_params / full_params) * 100
            savings_mb = (full_compressed - svd_compressed) / 1e6
            
            fits_note = ""
            if label.startswith("v=16384"):
                # Check if full model fits with this embedding
                body_est = 10_000_000 * 0.68  # ~6.8MB for transformer body
                total = body_est + svd_compressed + 45000  # + code
                fits = total < 16_000_000
                fits_note = f" | Total artifact: {total/1e6:.2f}MB {'✓' if fits else '✗'}"
            
            print(f"    SVD rank={rank:>3}: {svd_params:>12,} params "
                  f"({svd_compressed/1e6:.2f}MB, "
                  f"saves {savings_pct:.0f}% = {savings_mb:.2f}MB){fits_note}")


def test_svd_embedding():
    """Verify SVD embedding works correctly."""
    print(f"\n{'='*70}")
    print("FUNCTIONAL TESTS")
    print(f"{'='*70}")
    
    vocab, dim, rank = 16384, 512, 64
    
    # Basic test
    print("\n  Test 1: Basic forward pass")
    emb = SVDEmbedding(vocab, dim, rank)
    ids = torch.randint(0, vocab, (2, 32))
    out = emb(ids)
    print(f"    Input: {ids.shape} → Output: {out.shape}")
    assert out.shape == (2, 32, dim)
    print(f"    ✓ Shape correct")
    
    # Weight reconstruction
    print("\n  Test 2: Weight reconstruction (for tied embeddings)")
    W = emb.weight
    print(f"    Reconstructed weight shape: {W.shape}")
    assert W.shape == (vocab, dim)
    
    # Check that forward and weight are consistent
    manual = emb.weight[ids[0, 0]]
    forward = out[0, 0]
    diff = (manual - forward).abs().max().item()
    print(f"    Max diff (forward vs weight[id]): {diff:.8f}")
    assert diff < 1e-5, f"Inconsistency: {diff}"
    print(f"    ✓ Forward and weight are consistent")
    
    # Gradient flow
    print("\n  Test 3: Gradient flow")
    loss = out.sum()
    loss.backward()
    assert emb.U.grad is not None
    assert emb.S.grad is not None
    assert emb.V.grad is not None
    print(f"    U.grad norm: {emb.U.grad.norm():.4f}")
    print(f"    S.grad norm: {emb.S.grad.norm():.4f}")
    print(f"    V.grad norm: {emb.V.grad.norm():.4f}")
    print(f"    ✓ All gradients flow")
    
    # Tied embedding logit computation
    print("\n  Test 4: Tied embedding logit computation")
    hidden = torch.randn(2, 32, dim)
    logits = F.linear(hidden, emb.weight)  # [2, 32, vocab]
    print(f"    Logits shape: {logits.shape}")
    assert logits.shape == (2, 32, vocab)
    logits.sum().backward()
    print(f"    ✓ Backward through tied logits works")
    
    # With BigramHash
    print("\n  Test 5: SVDEmbeddingWithHash")
    emb_hash = SVDEmbeddingWithHash(vocab, dim, rank, bigram_vocab=2048)
    out_hash = emb_hash(ids)
    print(f"    Output shape: {out_hash.shape}")
    params = sum(p.numel() for p in emb_hash.parameters())
    print(f"    Total params: {params:,}")
    W_hash = emb_hash.weight
    print(f"    Weight shape: {W_hash.shape}")
    print(f"    ✓ All correct")
    
    # Compare param count
    full_emb = nn.Embedding(vocab, dim)
    full_params = sum(p.numel() for p in full_emb.parameters())
    svd_params = sum(p.numel() for p in emb.parameters())
    print(f"\n  Parameter comparison:")
    print(f"    Full nn.Embedding: {full_params:,}")
    print(f"    SVDEmbedding:      {svd_params:,}")
    print(f"    Savings:           {(1-svd_params/full_params)*100:.1f}%")
    
    print(f"\n  ✓ ALL TESTS PASSED")


def mini_training_comparison():
    """Compare training with full vs SVD embedding."""
    print(f"\n{'='*70}")
    print("MINI TRAINING: Full Embedding vs SVD Embedding")
    print(f"{'='*70}")
    
    import time
    
    vocab = 4096  # big enough to see the difference
    dim = 128
    rank = 32
    steps = 100
    B, T = 8, 32
    
    # Synthetic data
    torch.manual_seed(42)
    data = torch.randint(0, vocab, (200, T + 1))
    for i in range(0, 200, 3):
        plen = torch.randint(3, 8, (1,)).item()
        pat = torch.randint(0, vocab, (plen,))
        for j in range(T + 1):
            data[i, j] = pat[j % plen]
    
    class MiniLM(nn.Module):
        def __init__(self, vocab, dim, use_svd=False, rank=32):
            super().__init__()
            if use_svd:
                self.emb = SVDEmbedding(vocab, dim, rank)
            else:
                self.emb = nn.Embedding(vocab, dim)
            self.ln = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * 2)
            self.fc2 = nn.Linear(dim * 2, dim)
            self.norm = nn.LayerNorm(dim)
            self.vocab = vocab
            self.use_svd = use_svd
        
        def forward(self, idx, targets=None):
            x = self.ln(self.emb(idx))
            x = x + self.fc2(F.gelu(self.fc1(x)))
            x = self.norm(x)
            
            if self.use_svd:
                logits = F.linear(x, self.emb.weight)
            else:
                logits = F.linear(x, self.emb.weight)
            
            if targets is not None:
                return F.cross_entropy(logits.reshape(-1, self.vocab), targets.reshape(-1))
            return logits
    
    results = {}
    for use_svd, label in [(False, "Full Embedding"), (True, f"SVD rank={rank}")]:
        torch.manual_seed(42)
        model = MiniLM(vocab, dim, use_svd, rank)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        
        params = sum(p.numel() for p in model.parameters())
        losses = []
        
        t0 = time.time()
        for step in range(steps):
            idx = torch.randint(0, 200, (B,))
            batch = data[idx]
            loss = model(batch[:, :-1], batch[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        t1 = time.time()
        
        final = sum(losses[-10:]) / 10
        results[label] = {'final': final, 'min': min(losses), 'params': params, 'time': t1-t0}
        
        print(f"\n  {label}:")
        print(f"    Params: {params:,}")
        print(f"    Final loss: {final:.4f}")
        print(f"    Min loss: {min(losses):.4f}")
        print(f"    Time: {t1-t0:.1f}s")
    
    full = results["Full Embedding"]
    svd = results[f"SVD rank={rank}"]
    delta = svd['final'] - full['final']
    param_savings = (1 - svd['params'] / full['params']) * 100
    
    print(f"\n  COMPARISON:")
    print(f"    Loss delta: {delta:+.4f} ({'SVD WINS' if delta < -0.01 else 'SVD LOSES' if delta > 0.01 else 'TIE'})")
    print(f"    Param savings: {param_savings:.1f}%")
    print(f"    SVD quality per param is {'BETTER' if delta < 0 else 'WORSE'}")


if __name__ == "__main__":
    analyze_savings()
    test_svd_embedding()
    mini_training_comparison()
    
    print(f"""

{'='*70}
INTEGRATION INTO train_gpt.py
{'='*70}

1. Replace nn.Embedding with SVDEmbedding:

   # OLD:
   self.tok_emb = nn.Embedding(vocab_size, model_dim)
   
   # NEW:
   svd_rank = int(os.environ.get("SVD_RANK", 64))
   if svd_rank > 0 and vocab_size > 4096:
       self.tok_emb = SVDEmbedding(vocab_size, model_dim, rank=svd_rank)
   else:
       self.tok_emb = nn.Embedding(vocab_size, model_dim)

2. Tied embeddings (logit projection) — works automatically:
   
   # This already works because SVDEmbedding.weight returns [vocab, dim]
   logits = F.linear(x, self.tok_emb.weight)

3. Quantization pipeline — needs one change:
   
   # During quantization, reconstruct the full weight matrix FIRST,
   # then quantize the full matrix normally:
   if isinstance(model.tok_emb, SVDEmbedding):
       full_weight = model.tok_emb.weight.detach()
       # Quantize full_weight as a normal 2D tensor
       # This is what gets saved in the artifact
   
   # ALTERNATIVELY: quantize U, S, V separately for better compression:
   # Save U as int6 [16384, 64], S as float16 [64], V as int6 [64, 512]
   # Reconstruct at load time: E = (dequant(U) * S) @ dequant(V)
   # This is smaller than quantizing the full [16384, 512] matrix

4. Add to Hyperparameters:
   svd_rank = int(os.environ.get("SVD_RANK", 64))

5. Optimizer: SVDEmbedding params go in AdamW group (not Muon).
   U, S, V are all small enough for Adam.
""")
