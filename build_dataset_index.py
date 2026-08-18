"""
Build and cache dataset index for fast training startup.

This script pre-computes the dataset index (file paths, shapes, window counts)
and saves it to disk. The training script can then load this cached index
instead of re-scanning all files on every run, saving ~500+ seconds per job.

Usage:
    python build_dataset_index.py \
        --data_dir /scratch/hakati/spectrograms_npy \
        --output_file dataset_index_cache.pkl \
        --num_workers 8
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_file_info(file_path: Path) -> Tuple[str, Tuple]:
    """
    Load file metadata (path, shape).
    
    Returns:
        (file_path_str, shape_tuple)
    """
    try:
        if file_path.suffix == ".npz":
            with np.load(file_path) as z:
                key = list(z.files)[0]
                shape = z[key].shape
        else:  # .npy
            arr = np.load(file_path, mmap_mode="r")
            shape = arr.shape
        return (str(file_path), shape)
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}")
        return None


def build_index_parallel(
    data_dir: Path,
    pattern: str = "*.np[yz]",
    max_workers: int = 8,
) -> List[Tuple[str, Tuple]]:
    """
    Build dataset index in parallel.
    
    Args:
        data_dir: Directory containing .npy/.npz files
        pattern: Glob pattern for files
        max_workers: Number of parallel workers
    
    Returns:
        List of (file_path, shape) tuples
    """
    files = sorted(data_dir.glob(pattern))
    print(f"Found {len(files)} spectrogram files in {data_dir}")
    
    if len(files) == 0:
        raise ValueError(f"No .npy/.npz files found in {data_dir}")
    
    index = []
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(get_file_info, f): f for f in files}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is not None:
                index.append(result)
            
            if completed % 500 == 0:
                elapsed = time.time() - start
                print(f"  Indexed {completed}/{len(files)} files ({elapsed:.1f}s)")
    
    elapsed = time.time() - start
    print(f"✓ Indexed {len(index)}/{len(files)} files in {elapsed:.1f}s "
          f"({elapsed/len(files)*1000:.1f}ms per file)")
    
    return index


def compute_sample_index(
    file_index: List[Tuple[str, Tuple]]
) -> List[Tuple[str, int]]:
    """
    Convert file-level index to sample-level index.
    
    Each file may contain multiple windows along axis 1.
    
    Args:
        file_index: List of (file_path, shape)
    
    Returns:
        List of (file_path, window_idx) for each sample
    """
    sample_index = []
    
    for file_path, shape in file_index:
        if len(shape) == 4 and shape[0] == 128:
            # (128, W, 224, 224) - multiple windows per file
            n_windows = shape[1]
            for w in range(n_windows):
                sample_index.append((file_path, w))
        elif len(shape) == 3 and shape[0] == 128 and shape[1] == 224 and shape[2] == 224:
            # (128, 224, 224) - single window per file
            sample_index.append((file_path, -1))
        else:
            print(f"⚠️  Skipping {file_path}: unexpected shape {shape}")
    
    print(f"✓ Total samples: {len(sample_index)}")
    return sample_index


def save_index(
    index: List[Tuple[str, int]],
    output_file: Path,
) -> None:
    """Save index to disk."""
    with open(output_file, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"✓ Saved index to {output_file} ({size_mb:.1f} MB)")


def load_index(cache_file: Path) -> List[Tuple[str, int]]:
    """Load cached index from disk."""
    with open(cache_file, "rb") as f:
        index = pickle.load(f)
    print(f"✓ Loaded cached index from {cache_file} ({len(index)} samples)")
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Build and cache dataset index for fast training startup"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing .npy/.npz spectrogram files",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default="dataset_index_cache.pkl",
        help="Path to save cached index",
    )
    parser.add_argument(
        "--pattern",
        default="*.np[yz]",
        help="Glob pattern for files (default: *.np[yz])",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild index even if cache exists",
    )
    
    args = parser.parse_args()
    
    # Check if cache exists
    if args.output_file.exists() and not args.force:
        print(f"Cache exists: {args.output_file}")
        print("Use --force to rebuild")
        return
    
    print("=" * 80)
    print("Building Dataset Index Cache")
    print("=" * 80)
    print(f"Data dir        : {args.data_dir}")
    print(f"Output file     : {args.output_file}")
    print(f"Workers         : {args.num_workers}")
    print("=" * 80)
    
    # Build file-level index in parallel
    file_index = build_index_parallel(
        args.data_dir,
        pattern=args.pattern,
        max_workers=args.num_workers,
    )
    
    # Convert to sample-level index
    print("\nConverting to sample-level index...")
    sample_index = compute_sample_index(file_index)
    
    # Save to disk
    print("\nSaving index...")
    save_index(sample_index, args.output_file)
    
    print("\n" + "=" * 80)
    print("✓ Done! Use this cache in pretrain_mae.py:")
    print(f"  python pretrain_mae.py --index_cache {args.output_file} ...")
    print("=" * 80)


if __name__ == "__main__":
    main()
