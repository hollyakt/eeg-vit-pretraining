#!/usr/bin/env python3
import os, sys
from pathlib import Path
import numpy as np
SRC = Path("/scratch/hakati/spectrograms")
DST = Path("/scratch/hakati/spectrograms_npy")
KEY = "spectrograms"
def main():
    num_shards = int(sys.argv[1]); shard_id = int(sys.argv[2])
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.npz"))[shard_id::num_shards]
    wrote = skipped = failed = 0
    for f in files:
        out = DST / (f.stem + ".npy")
        if out.exists(): skipped += 1; continue
        tmp = DST / (f.stem + ".tmp.npy")
        try:
            with np.load(f) as z:
                arr = np.ascontiguousarray(z[KEY])
            np.save(tmp, arr)
            os.replace(tmp, out)
            wrote += 1
        except Exception as e:
            if tmp.exists():
                try: tmp.unlink()
                except OSError: pass
            print(f"FAIL {f.name}: {e}", flush=True); failed += 1
    print(f"shard {shard_id}/{num_shards}: wrote {wrote}, skipped {skipped}, failed {failed}", flush=True)
if __name__ == "__main__": main()
