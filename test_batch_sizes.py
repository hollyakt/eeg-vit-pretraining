#!/usr/bin/env python
"""
Test different batch sizes to find optimal throughput and memory usage.

This script runs short training runs with different batch sizes and measures:
  - GPU memory usage
  - Throughput (samples/sec)
  - Training loss

Usage:
    python test_batch_sizes.py --data_dir /scratch/hakati/spectrograms_npy \
                               --batch_sizes 64 128 256 512 \
                               --test_steps 50

Results are saved to batch_size_benchmark.json for comparison.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.cuda
from torch.optim import AdamW
from torch.utils.data import DataLoader

import vision_transformer as vits
from pretrain_mae import EEGSpectrogramDataset, GroupedShuffleSampler, MAEPretrainer


def measure_throughput(data_loader, model, optimizer, device, num_steps=50):
    """Run model and measure throughput."""
    model.train()
    
    # Warmup step
    for batch_idx, (x,) in enumerate(data_loader):
        if batch_idx >= 1:
            break
        x = x.to(device)
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
        
        loss = model(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Measure throughput
    torch.cuda.synchronize()
    start = time.time()
    total_samples = 0
    total_loss = 0
    
    for batch_idx, (x,) in enumerate(data_loader):
        if batch_idx >= num_steps:
            break
        
        x = x.to(device)
        loss = model(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_samples += x.shape[0]
        total_loss += loss.item()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Get peak memory
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
    throughput = total_samples / elapsed
    avg_loss = total_loss / min(num_steps, batch_idx + 1)
    
    return {
        "throughput_samples_per_sec": throughput,
        "peak_memory_gb": peak_mem_gb,
        "avg_loss": avg_loss,
        "elapsed_sec": elapsed,
        "total_samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Test batch sizes for optimal throughput")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--test_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index_cache", type=str, default=None)
    parser.add_argument("--output", type=str, default="batch_size_benchmark.json")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = EEGSpectrogramDataset(
        args.data_dir,
        augment=False,
        normalize=True,
        index_cache=args.index_cache,
    )
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Build model
    print("Building ViT-Small model...")
    encoder = vits.vit_small(img_size=224, patch_size=16, in_chans=128, num_classes=0)
    model = MAEPretrainer(encoder, embed_dim=384, mask_ratio=0.75, norm_pix_loss=True)
    model = model.to(device)
    print(f"✓ Model ready: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    
    results: Dict[int, Dict] = {}
    
    print("\n" + "="*70)
    print("BATCH SIZE BENCHMARK")
    print("="*70)
    
    for batch_size in args.batch_sizes:
        print(f"\nTesting batch_size={batch_size}...")
        
        try:
            # Create sampler and loader
            sampler = GroupedShuffleSampler(dataset.index, seed=args.seed)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )
            
            # Create optimizer
            optimizer = AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
            
            # Measure
            metrics = measure_throughput(loader, model, optimizer, device, num_steps=args.test_steps)
            results[batch_size] = metrics
            
            print(f"  Throughput: {metrics['throughput_samples_per_sec']:.0f} samples/sec")
            print(f"  Peak Memory: {metrics['peak_memory_gb']:.2f} GB")
            print(f"  Avg Loss: {metrics['avg_loss']:.4f}")
            print(f"  Time: {metrics['elapsed_sec']:.1f}s for {metrics['total_samples']} samples")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ❌ OOM - batch size too large for this GPU")
                results[batch_size] = {"error": "out_of_memory"}
            else:
                raise
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_batch = max(valid_results, key=lambda k: valid_results[k]["throughput_samples_per_sec"])
        best_throughput = valid_results[best_batch]["throughput_samples_per_sec"]
        
        print(f"\nBest batch size: {best_batch} ({best_throughput:.0f} samples/sec)")
        print(f"Speedup vs batch=128: {best_throughput / valid_results.get(128, {}).get('throughput_samples_per_sec', 1):.2f}x")
        
        print("\nDetailed results:")
        print(f"{'Batch Size':<12} {'Throughput':<20} {'Memory (GB)':<15} {'Loss':<10}")
        print("-" * 60)
        for batch in sorted(valid_results.keys()):
            r = valid_results[batch]
            print(f"{batch:<12} {r['throughput_samples_per_sec']:<20.0f} {r['peak_memory_gb']:<15.2f} {r['avg_loss']:<10.4f}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
