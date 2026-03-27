"""
Re-tokenize FineWeb data with the 16384-vocab tokenizer.
Creates binary shards in the same format as the original.
"""
import numpy as np
import sentencepiece as spm
import glob
import os
import time
from pathlib import Path

# Load tokenizers
sp_old = spm.SentencePieceProcessor()
sp_old.Load("data/tokenizers/fineweb_1024_bpe.model")

sp_new = spm.SentencePieceProcessor()
sp_new.Load("data/tokenizers/fineweb_16384_bpe.model")

# Output directory
out_dir = "data/datasets/fineweb10B_sp16384"
os.makedirs(out_dir, exist_ok=True)

shard_files = sorted(glob.glob("data/datasets/fineweb10B_sp1024/fineweb_*.bin"))

print(f"Re-tokenizing {len(shard_files)} shards with vocab=16384...")
t0 = time.time()

for shard_path in shard_files:
    shard_name = Path(shard_path).name
    out_path = os.path.join(out_dir, shard_name)

    if os.path.exists(out_path):
        print(f"  Skipping {shard_name} (already exists)")
        continue

    print(f"  Processing {shard_name}...")
    st = time.time()

    # Read old tokens
    header = np.fromfile(shard_path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    old_tokens = np.fromfile(shard_path, dtype="<u2", count=num_tokens,
                             offset=256 * 4)

    # Decode to text in chunks, then re-encode
    chunk_size = 5000
    new_tokens_list = []

    for i in range(0, len(old_tokens), chunk_size):
        chunk = old_tokens[i:i+chunk_size].tolist()
        text = sp_old.Decode(chunk)
        new_ids = sp_new.Encode(text)
        new_tokens_list.extend(new_ids)

        if i % 500000 == 0 and i > 0:
            elapsed = time.time() - st
            print(f"    {i:,}/{num_tokens:,} tokens ({elapsed:.0f}s)")

    new_tokens = np.array(new_tokens_list, dtype=np.uint16)

    # Verify all tokens are < 16384
    assert new_tokens.max() < 16384, f"Token {new_tokens.max()} >= 16384!"

    # Write new shard
    new_header = np.zeros(256, dtype=np.int32)
    new_header[0] = 20240520  # magic
    new_header[1] = 1         # version
    new_header[2] = len(new_tokens)

    with open(out_path, "wb") as f:
        f.write(new_header.tobytes())
        f.write(new_tokens.tobytes())

    ratio = len(old_tokens) / len(new_tokens) if len(new_tokens) > 0 else 0
    elapsed = time.time() - st
    print(f"    {len(old_tokens):,} -> {len(new_tokens):,} tokens "
          f"(ratio: {ratio:.2f}x, ~{ratio * 2.44:.1f} bytes/token, {elapsed:.0f}s)")

total_elapsed = time.time() - t0
print(f"\nDone! New shards in: {out_dir} ({total_elapsed:.0f}s)")
