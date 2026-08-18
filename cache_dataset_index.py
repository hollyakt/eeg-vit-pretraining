"""
Build and cache dataset index for fast training startup.

This script pre-computes the dataset index once, avoiding the 562s indexing
delay on every training run. The index is cached to disk and reloaded instantly.

Usage:
    python cache_dataset_index.py --data_dir /scratch/hakati/spectrograms_npy

Then in pretrain_mae.py, the dataset loads from cache instead of re-indexing.
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


def build_index(data_dir: str, npz_pattern: str = "*.np[yz]") -> List[Tuple[str, int]]:
    """
    Build dataset index mapping (file_path, window_idx) for all spectrograms.
    
    Args:
        data_dir: Directory containing .npy/.npz files
        npz_pattern: Glob pattern to match spectrogram files
        
    Returns:
        List of (file_path, window_idx) tuples
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(npz_pattern))
    
    if not files:
        raise ValueError(f"No .npy/.npz files found in {data_dir} matching {npz_pattern}")
    
    print(f"Building index for {len(files)} files in {data_dir}")
    index: List[Tuple[str, int]] = []
    
    start = time.time()
    for f in tqdm(files, desc="Indexing files"):
        try:
            # Get shape without loading data
            if f.suffix == ".npz":
                with np.load(f) as z:
                    key = list(z.files)[0]
                    shape = z[key].shape
            else:  # .npy
                shape = np.load(f, mmap_mode="r").shape
            
            # Handle both single-window and multi-window files
            if len(shape) == 4 and shape[0] == 128:
                # Multi-window: (128, W, 224, 224) -> W windows
                for w in range(shape[1]):
                    index.append((str(f), w))
            elif len(shape) == 3 and shape[0] == 128 and shape[-1] == 224 and shape[-2] == 224:
                # Single window: (128, 224, 224)
                index.append((str(f), -1))
            else:
                print(f"⚠️  Skipping {f.name}: unexpected shape {shape}")
        
        except Exception as e:
            print(f"⚠️  Error indexing {f.name}: {e}")
            continue
    
    elapsed = time.time() - start
    print(f"\n✓ Indexed {len(index)} samples in {elapsed:.1f}s")
    print(f"  Average: {len(files)/elapsed:.0f} files/sec")
    
    return index


def save_index(index: List[Tuple[str, int]], cache_path: str) -> None:
    """Save index to pickle file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    
    size_mb = cache_path.stat().st_size / (1024 ** 2)
    print(f"✓ Saved index to {cache_path} ({size_mb:.1f} MB)")


def load_index(cache_path: str) -> List[Tuple[str, int]]:
    """Load index from pickle file."""
    cache_path = Path(cache_path)
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Index cache not found: {cache_path}")
    
    with open(cache_path, "rb") as f:
        index = pickle.load(f)
    
    print(f"✓ Loaded index from cache: {len(index)} samples")
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Build and cache dataset index for fast training startup"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing .npy/.npz spectrogram files"
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="dataset_index.pkl",
        help="Path to save cached index (default: dataset_index.pkl)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify cached index by loading and comparing file counts"
    )
    
    args = parser.parse_args()
    
    # Build index
    print("\n" + "="*70)
    print("Building dataset index (first time only)")
    print("="*70)
    index = build_index(args.data_dir)
    
    # Save to cache
    save_index(index, args.cache_path)
    
    # Verify if requested
    if args.verify:
        print("\nVerifying cached index...")
        loaded = load_index(args.cache_path)
        assert len(loaded) == len(index), f"Mismatch: {len(loaded)} != {len(index)}"
        print("✓ Index verification passed")
    
    print("\n" + "="*70)
    print("Next time, index will load instantly from cache!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
