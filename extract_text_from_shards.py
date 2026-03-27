"""Extract raw text from FineWeb binary shards for tokenizer training.
Adds newlines every ~500 tokens to create proper sentences for SentencePiece."""
import numpy as np
import sentencepiece as spm
import glob
import time

sp = spm.SentencePieceProcessor()
sp.Load("data/tokenizers/fineweb_1024_bpe.model")

output_file = "fineweb_raw_text_for_tokenizer.txt"
max_bytes = 1_000_000_000  # 1GB

total_bytes = 0
shard_files = sorted(glob.glob("data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))
print(f"Found {len(shard_files)} shards")
t0 = time.time()

with open(output_file, "w", encoding="utf-8") as out:
    for shard_path in shard_files:
        print(f"Processing {shard_path}...")
        header = np.fromfile(shard_path, dtype="<i4", count=256)
        num_tokens = int(header[2])
        tokens = np.fromfile(shard_path, dtype="<u2", count=num_tokens, offset=256 * 4)

        # Decode in chunks of 500 tokens, add newline after each chunk
        chunk_size = 500
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i+chunk_size].tolist()
            text = sp.Decode(chunk)
            out.write(text)
            out.write("\n")
            total_bytes += len(text.encode("utf-8")) + 1

            if i % 500000 == 0 and i > 0:
                elapsed = time.time() - t0
                print(f"  {total_bytes / 1e6:.0f}MB extracted ({elapsed:.0f}s)")

            if total_bytes >= max_bytes:
                break

        if total_bytes >= max_bytes:
            break
        print(f"  Total: {total_bytes / 1e6:.0f}MB")

elapsed = time.time() - t0
print(f"\nDone: {total_bytes / 1e6:.0f}MB written to {output_file} ({elapsed:.0f}s)")
