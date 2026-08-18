"""
Self-Supervised Pre-training Script for Vision Transformer on EEG Spectrograms

This script implements Masked Autoencoder (MAE) pre-training for the ViT-Small model
on 128-channel EEG spectrograms. The pre-trained backbone will be frozen during
fine-tuning and only the regression head will be trained, as per the methodology.

Architecture: ViT-Small with 128-channel input (multi-channel spectrograms)
Optimizer: AdamW
Learning Rate Schedule: Cosine annealing with warmup
Loss: MSE (reconstruction)
"""

import math
import os
import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
import gc
import random

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import vision_transformer as vits
from torchvision import transforms


# ============================================================================
# DATA LOADING
# ============================================================================

class EEGSpectrogramDataset(Dataset):
    """
    Lazy dataset for EEG spectrogram .npz files.

    Instead of preloading every spectrogram into RAM, this indexes
    (file, window) pairs up front and loads each window from disk on demand,
    so memory stays flat regardless of how many files/releases are used.

    Each item is a FloatTensor of shape (128, 224, 224).
    """

    def __init__(self, data_dir: str, npz_pattern: str = "*.npz",
                 max_samples: Optional[int] = None,
                 augment: bool = False,
                 normalize: bool = True):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.normalize = normalize

        npz_files = sorted(self.data_dir.glob(npz_pattern))
        print(f"Found {len(npz_files)} .npz files in {data_dir}")
        if len(npz_files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        self.index = []
        for npz_file in npz_files:
            try:
                shape = self._spectrograms_shape(npz_file)
            except Exception as e:
                print(f"Error indexing {npz_file}: {e}")
                continue
            if len(shape) == 4 and shape[0] == 128:
                for w in range(shape[1]):
                    self.index.append((str(npz_file), w))
            elif len(shape) == 3 and shape[-1] == 224 and shape[-2] == 224:
                self.index.append((str(npz_file), -1))
            else:
                print(f"Skipping {npz_file}: unexpected shape {shape}")

        if max_samples:
            self.index = self.index[:max_samples]

        print(f"Indexed {len(self.index)} spectrogram samples (lazy loading)")
        if len(self.index) == 0:
            raise ValueError("No spectrogram samples found")

        self._cache_path = None
        self._cache_arr = None

    @staticmethod
    @staticmethod
       def _spectrograms_shape(npz_path):
           return tuple(np.load(npz_path, mmap_mode="r").shape)

    def __len__(self):
        return len(self.index)

    def _load_array(self, npz_path):
        if npz_path != self._cache_path:
           self._cache_arr = np.load(npz_path, mmap_mode="r")
            self._cache_path = npz_path
        return self._cache_arr

    def __getitem__(self, idx):
        npz_path, w = self.index[idx]
        arr = self._load_array(npz_path)

        if w < 0:
            spec = np.asarray(arr, dtype=np.float32)
        else:
            spec = arr[:, w, :, :].astype(np.float32)

        if spec.ndim != 3 or spec.shape[0] != 128:
            raise ValueError(f"Unexpected spectrogram shape: {spec.shape}, expected (128, 224, 224)")

        if self.normalize:
            smin, smax = spec.min(), spec.max()
            spec = (spec - smin) / (smax - smin) if smax > smin else np.zeros_like(spec)

        if self.augment:
            spec = self._augment_spectrogram(spec)

        return torch.from_numpy(spec).float()

    @staticmethod
    def _augment_spectrogram(spec: np.ndarray) -> np.ndarray:
        spec = spec.copy()
        spec = np.clip(spec * np.random.uniform(0.9, 1.1), 0, 1)
        shift = np.random.randint(-2, 3)
        if shift != 0:
            spec = np.roll(spec, shift, axis=-1)
        spec = np.clip(spec + np.random.normal(0, 0.01, spec.shape), 0, 1)
        return spec


# ============================================================================
# MASKED AUTOENCODER (MAE) COMPONENTS
# ============================================================================

class MAEHead(nn.Module):
    """Reconstruction head for masked autoencoder pre-training."""
    
    def __init__(self, embed_dim: int = 384, patch_size: int = 16, 
                 in_chans: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        
        # Number of patches: (224 / 16)^2 = 196 patches
        self.n_patches = (224 // patch_size) ** 2
        
        # Decoder: embed_dim -> patch_size^2 * in_chans
        patch_dim = patch_size * patch_size * in_chans
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) - encoder output
        
        Returns:
            (B, in_chans, H, W) - reconstructed image
        """
        B, N, D = x.shape
        
        # Decode each token
        patches = self.decoder(x)  # (B, N, patch_dim)
        
        # Reshape patches back to image
        patches = patches.view(B, N, self.in_chans, self.patch_size, self.patch_size)
        # (B, N, in_chans, patch_size, patch_size)
        
        # Reorganize patches into image grid
        n_patches_side = 224 // self.patch_size
        patches = patches.view(B, n_patches_side, n_patches_side, 
                             self.in_chans, self.patch_size, self.patch_size)
        # (B, n_patches_side, n_patches_side, in_chans, patch_size, patch_size)
        
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        # (B, in_chans, n_patches_side, patch_size, n_patches_side, patch_size)
        
        img = patches.view(B, self.in_chans, 224, 224)
        # (B, in_chans, 224, 224)
        
        return img


class MAEPretrainer(nn.Module):
    """Masked Autoencoder for pre-training ViT on EEG spectrograms."""
    
    def __init__(self, encoder: nn.Module, embed_dim: int = 384,
                 patch_size: int = 16, in_chans: int = 128,
                 mask_ratio: float = 0.75):
        """
        Args:
            encoder: Vision Transformer encoder
            embed_dim: Embedding dimension
            patch_size: Patch size
            in_chans: Input channels (128 for EEG)
            mask_ratio: Ratio of patches to mask (0.75 = mask 75%)
        """
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.n_patches = (224 // patch_size) ** 2
        self.n_keep = int(self.n_patches * (1 - mask_ratio))
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Reconstruction head
        self.mae_head = MAEHead(embed_dim, patch_size, in_chans)
    
    def _random_masking(self, x: torch.Tensor) -> tuple:
        """
        Random masking by shuffling.
        
        Args:
            x: (N, L, D) - L is the number of patches
        
        Returns:
            x_masked: (N, L, D) - with masked patches set to mask_token
            mask: (N, L) - binary mask (1 = masked, 0 = kept)
            ids_restore: indices to restore original order
        """
        N, L, D = x.shape
        len_keep = self.n_keep
        
        # Generate noise for shuffling
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise and get indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first N_keep patches (after shuffling)
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Create mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Gather kept patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        return x_masked, mask, ids_restore
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, in_chans, 224, 224) - input spectrograms
        
        Returns:
            loss: scalar
            pred: (B, in_chans, 224, 224) - predicted spectrograms
            target: (B, in_chans, 224, 224) - target spectrograms
        """
        # Forward through encoder to get patch embeddings
        # The encoder outputs shape (B, N, embed_dim) after patch embedding and ViT blocks
        B = x.shape[0]
        
        # Store original for reconstruction loss
        target = x.clone()
        
        # Get patch embeddings (B, N+1, embed_dim) where N+1 includes CLS token
        features = self.encoder(x, classify=False)  # Assuming this gives us CLS + patch features
        
        # Remove CLS token if present (keeping only patch tokens)
        if features.shape[1] == self.n_patches + 1:
            features = features[:, 1:, :]  # (B, N, embed_dim)
        
        # Random masking
        x_masked, mask, ids_restore = self._random_masking(features)
        
        # Replace masked patches with mask_token
        mask_tokens = self.mask_token.expand(B, self.n_patches, -1)
        x_masked_full = torch.cat([x_masked, mask_tokens[:, self.n_keep:, :]], dim=1)  # (B, N, D)
        x_masked_full = torch.gather(x_masked_full, dim=1, 
                                    index=ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        
        # Add CLS token back
        cls_token = features[:, :1, :] * 0  # Create zero CLS token
        x_with_cls = torch.cat([cls_token, x_masked_full], dim=1)
        
        # Reconstruct
        pred = self.mae_head(x_masked_full)  # (B, in_chans, 224, 224)
        
        # Compute reconstruction loss (only on masked patches)
        # For simplicity, we compute MSE over entire image
        loss = ((pred - target) ** 2).mean()
        
        return loss, pred, target, mask


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def fix_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.deterministic = True


def create_optimizer(model: nn.Module, lr: float = 1e-4, 
                    weight_decay: float = 0.05) -> AdamW:
    """Create AdamW optimizer."""
    return AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        eps=1e-8
    )


def create_scheduler(optimizer, num_epochs: int, num_steps_per_epoch: int,
                    warmup_epochs: int = 5):
    """Create cosine annealing scheduler with warmup."""
    total_steps = num_epochs * num_steps_per_epoch
    warmup_steps = warmup_epochs * num_steps_per_epoch
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / 
                  float(max(1, total_steps - warmup_steps)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int, 
                   loss: float, checkpoint_dir: str):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"pretrain_ckpt_ep{epoch:04d}.pth"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer, scheduler):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resume from epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.6f}")
    
    return checkpoint['epoch']


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model: nn.Module, data_loader: DataLoader, 
               optimizer, scheduler, device: str, epoch: int, 
               log_interval: int = 100) -> float:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = len(data_loader)
    
    for batch_idx, batch in enumerate(data_loader):
        # Get batch
        spectrograms = batch.to(device)  # (B, 128, 224, 224)
        
        # Forward pass
        loss, pred, target, mask = model(spectrograms)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.6f} (Avg: {avg_loss:.6f}) "
                  f"LR: {lr:.2e}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model: nn.Module, data_loader: DataLoader, 
            device: str) -> float:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = len(data_loader)
    
    for batch_idx, batch in enumerate(data_loader):
        spectrograms = batch.to(device)
        
        loss, pred, target, mask = model(spectrograms)
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    # ========================================================================
    # ARGUMENTS
    # ========================================================================
    
    parser = argparse.ArgumentParser('Pre-training script for ViT on EEG spectrograms')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing .npz spectrogram files')
    parser.add_argument('--val_data_dir', type=str, default=None,
                       help='Directory for validation data (optional)')
    parser.add_argument('--output_dir', type=str, default='./pretrain_checkpoints',
                       help='Output directory for checkpoints')
    
    # Model
    parser.add_argument('--model', type=str, default='vit_small',
                       help='Model architecture (vit_tiny, vit_small, vit_base)')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size')
    parser.add_argument('--in_chans', type=int, default=128,
                       help='Number of input channels (128 for EEG)')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Ratio of patches to mask')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Warmup epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Checkpoint saving interval')
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    fix_random_seeds(args.seed)
    device = torch.device(args.device)
    
    print("="*80)
    print("EEG ViT Pre-training")
    print("="*80)
    print(f"Timestamp: {datetime.datetime.now()}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Mask ratio: {args.mask_ratio}")
    print("="*80)
    
    # ========================================================================
    # DATA
    # ========================================================================
    
    print("\nLoading training data...")
    train_dataset = EEGSpectrogramDataset(
        args.data_dir,
        augment=True,
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    if args.val_data_dir:
        print("Loading validation data...")
        val_dataset = EEGSpectrogramDataset(
            args.val_data_dir,
            augment=False,
            normalize=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        val_loader = None
    
    # ========================================================================
    # MODEL
    # ========================================================================
    
    print(f"\nBuilding model: {args.model}")
    
    # Create encoder
    if args.model == 'vit_tiny':
        encoder = vits.vit_tiny(
            patch_size=args.patch_size,
            img_size=[224],
            in_chans=args.in_chans,
            num_classes=0  # No classification head
        )
    elif args.model == 'vit_small':
        encoder = vits.vit_small(
            patch_size=args.patch_size,
            img_size=[224],
            in_chans=args.in_chans,
            num_classes=0
        )
    elif args.model == 'vit_base':
        encoder = vits.vit_base(
            patch_size=args.patch_size,
            img_size=[224],
            in_chans=args.in_chans,
            num_classes=0
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Create MAE model
    embed_dim = encoder.num_features
    model = MAEPretrainer(
        encoder=encoder,
        embed_dim=embed_dim,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        mask_ratio=args.mask_ratio
    )
    
    model = model.to(device)
    
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.epochs, len(train_loader), 
                                args.warmup_epochs)
    
    start_epoch = 0
    
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\nStarting pre-training...")
    print("="*80)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                               device, epoch, args.log_interval)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, device)
            print(f"Epoch {epoch}: Val Loss = {val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, 
                              val_loss, args.output_dir)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, 
                          train_loss, args.output_dir)
    
    print("\n" + "="*80)
    print("Pre-training completed!")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("="*80)
    
    # Save final model
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    final_path = Path(args.output_dir) / "pretrain_final.pth"
    torch.save(model.encoder.state_dict(), final_path)
    print(f"Final encoder saved to: {final_path}")


if __name__ == '__main__':
    main()
