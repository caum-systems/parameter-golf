"""2000-step comparison: BASELINE vs VALUE_RESIDUAL"""
import sys, os, time, math
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ['DATA_PATH'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'datasets', 'fineweb10B_sp1024')
from train_gpt_depthrecur import GPT, Hyperparameters, load_data_shard, load_validation_tokens
from pathlib import Path

args = Hyperparameters()
device = 'cuda'
seq_len = 512; steps = 2000; batch = 8

shard = load_data_shard(Path(args.data_path) / 'fineweb_train_000000.bin')
val_tokens = load_validation_tokens(args.val_files, seq_len)
print(f"Data loaded. {val_tokens.numel():,} val tokens")

def make_model(vr):
    return GPT(
        vocab_size=1024, num_layers=12, model_dim=768, num_heads=12,
        num_kv_heads=4, mlp_mult=3, tie_embeddings=True,
        tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
        qk_gain_init=1.5, bigram_vocab_size=2048, bigram_dim=128,
        xsa_last_n=4, rope_dims=16, ln_scale=True,
        ve_enabled=True, ve_dim=128, ve_layers='10,11',
        num_unique_blocks=4, repeats=3, lora_rank=8,
        deep_sup_enabled=False, value_residual=vr,
    ).to(device).bfloat16()

def eval_bpb(model):
    model.eval()
    total_nll = 0.0; total_tok = 0
    n_eval = min(160, (val_tokens.numel()-1)//seq_len)
    with torch.inference_mode():
        for si in range(0, n_eval, batch):
            be = min(si+batch, n_eval)
            bs = be-si
            xb = torch.zeros(bs,seq_len,dtype=torch.int64,device=device)
            yb = torch.zeros(bs,seq_len,dtype=torch.int64,device=device)
            for i in range(bs):
                s=(si+i)*seq_len
                tok=val_tokens[s:s+seq_len+1].to(dtype=torch.int64,device=device)
                xb[i]=tok[:-1]; yb[i]=tok[1:]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model.forward_logits(xb)
            nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),yb.reshape(-1),reduction='none')
            total_nll+=nll.sum().item(); total_tok+=bs*seq_len
    return (total_nll/total_tok)/math.log(2.0)

for vr_label, vr_val in [('BASELINE',False),('VALUE_RESIDUAL',True)]:
    torch.manual_seed(42)
    model = make_model(vr_val)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    t0 = time.time()
    for step in range(steps):
        start = step*seq_len*batch
        tokens = shard[start:start+seq_len*batch+1].to(dtype=torch.int64,device=device)
        x=tokens[:-1].reshape(batch,seq_len); y=tokens[1:].reshape(batch,seq_len)
        opt.zero_grad()
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            loss=model(x,y)
        loss.backward(); opt.step()
        if (step+1) in [500,1000,1500,2000]:
            bpb = eval_bpb(model)
            model.train()
            print(f'  [{vr_label}] step={step+1} train_loss={loss.item():.4f} val_bpb={bpb:.4f} time={time.time()-t0:.1f}s')
    del model; torch.cuda.empty_cache()

print("\nDone!")
