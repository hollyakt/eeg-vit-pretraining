"""
Complete Pre-training Pipeline: From Raw EEG to Pre-trained Model

This script provides an end-to-end workflow:
1. Load raw EEG data
2. Preprocess with filtering and artifact removal
3. Create spectrograms (Model 1 or Model 2 format)
4. Run pre-training
5. Save pre-trained model ready for fine-tuning

Usage:
    python pretrain_pipeline.py --raw_data_dir ./raw_data/ --output_dir ./pretrained_models/
"""

import os
import argparse
import json
from pathlib import Path
import gc
import numpy as np
from typing import Optional, Dict, List
import subprocess
import sys

# Import preprocessing utilities
try:
    from preprocessing import preprocess_eeg_for_spectrograms
    from spects import create_individual_channel_spectrograms_dynamic
except ImportError:
    print("Warning: Could not import preprocessing utilities")
    print("Make sure preprocessing.py and spects.py are in the same directory")


# ============================================================================
# SPECTROGRAM CREATION (MODEL 2 FORMAT - CONTINUOUS WINDOWS)
# ============================================================================

def create_training_spectrograms_model2(
    eeg_data: np.ndarray,
    sfreq: int = 100,
    window_size: int = 3000,  # 30 seconds at 100 Hz
    output_dir: str = './processed_data/',
    filename_base: str = 'participant',
    verbose: bool = True
) -> Dict:
    """
    Create training spectrograms in Model 2 format (continuous windows).
    
    Args:
        eeg_data: Shape (n_channels, n_samples), should be 128 channels
        sfreq: Sampling frequency (100 Hz for EEG challenge)
        window_size: Window size in samples (3000 = 30s @ 100 Hz)
        output_dir: Where to save .npz files
        filename_base: Base name for output files
        verbose: Print progress
    
    Returns:
        Dictionary with processing info
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create spectrograms
    results = create_individual_channel_spectrograms_dynamic(
        eeg_data,
        fs=sfreq,
        window_size=window_size,
        target_size=(224, 224),
        overlap_percent=0.5,
        frequency_focus=True,
        max_freq=50,
        min_freq=0.5,
        verbose=verbose
    )
    
    # Extract spectrograms
    all_specs = results['all_spectrograms']  # List of channel spectrograms
    
    # Stack into (128, n_windows, 224, 224)
    n_channels = len(all_specs)
    n_windows = len(all_specs[0]) if all_specs else 0
    
    if verbose:
        print(f"Creating stacked array: ({n_channels}, {n_windows}, 224, 224)")
    
    spectrograms_stacked = np.zeros((n_channels, n_windows, 224, 224), dtype=np.float32)
    
    for ch_idx, ch_specs in enumerate(all_specs):
        for window_idx, spec in enumerate(ch_specs):
            spectrograms_stacked[ch_idx, window_idx, :, :] = spec
    
    # Save to .npz
    output_file = output_dir / f"{filename_base}.npz"
    np.savez(
        output_file,
        spectrograms=spectrograms_stacked
    )
    
    if verbose:
        print(f"Saved: {output_file}")
        print(f"Shape: {spectrograms_stacked.shape}")
    
    return {
        'output_file': str(output_file),
        'shape': spectrograms_stacked.shape,
        'n_channels': n_channels,
        'n_windows': n_windows,
        'metadata': results['metadata']
    }


# ============================================================================
# DATA PREPARATION UTILITIES
# ============================================================================

def organize_data_for_training(
    spectrograms_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Dict:
    """
    Organize spectrograms into train/val splits.
    
    Args:
        spectrograms_dir: Directory with .npz files
        output_dir: Where to save organized data
        train_ratio: Ratio of data for training
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with split information
    """
    
    np.random.seed(seed)
    
    spectrograms_dir = Path(spectrograms_dir)
    output_dir = Path(output_dir)
    
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .npz files
    npz_files = sorted(spectrograms_dir.glob('*.npz'))
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {spectrograms_dir}")
    
    # Split files
    n_train = int(len(npz_files) * train_ratio)
    np.random.shuffle(npz_files)
    
    train_files = npz_files[:n_train]
    val_files = npz_files[n_train:]
    
    # Copy files
    import shutil
    
    for npz_file in train_files:
        shutil.copy(npz_file, train_dir / npz_file.name)
    
    for npz_file in val_files:
        shutil.copy(npz_file, val_dir / npz_file.name)
    
    split_info = {
        'total_files': len(npz_files),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'train_ratio': train_ratio,
        'train_dir': str(train_dir),
        'val_dir': str(val_dir),
    }
    
    print("\n" + "="*80)
    print("DATA SPLIT SUMMARY")
    print("="*80)
    print(f"Total files: {split_info['total_files']}")
    print(f"Training files: {split_info['train_files']}")
    print(f"Validation files: {split_info['val_files']}")
    print(f"Train directory: {train_dir}")
    print(f"Val directory: {val_dir}")
    
    return split_info


# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def run_full_pipeline(
    raw_data_dir: Optional[str] = None,
    spectrograms_dir: Optional[str] = None,
    output_dir: str = './pretrained_models/',
    model: str = 'vit_small',
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    mask_ratio: float = 0.75,
    device: str = 'cuda',
    **kwargs
):
    """
    Run complete pre-training pipeline.
    
    Args:
        raw_data_dir: Directory with raw EEG .mat files (optional)
        spectrograms_dir: Directory with pre-computed .npz files (optional)
        output_dir: Where to save results
        model: ViT model size
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        mask_ratio: Masking ratio for MAE
        device: cuda or cpu
        **kwargs: Additional arguments
    """
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + "EEG ViT PRE-TRAINING PIPELINE".center(78) + "║")
    print("╚" + "="*78 + "╝\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ====================================================================
    # STAGE 1: PREPARE DATA
    # ====================================================================
    
    print("\n" + "="*80)
    print("STAGE 1: PREPARING DATA")
    print("="*80)
    
    if spectrograms_dir:
        print(f"Using pre-computed spectrograms: {spectrograms_dir}")
        processed_dir = Path(spectrograms_dir)
    
    elif raw_data_dir:
        print(f"Processing raw EEG data from: {raw_data_dir}")
        processed_dir = output_dir / 'spectrograms'
        
        # TODO: Implement raw EEG processing
        # This would involve:
        # 1. Loading .mat files
        # 2. Preprocessing (filtering, artifact removal)
        # 3. Creating spectrograms
        
        print("Raw EEG processing not yet implemented.")
        print("Please provide pre-computed spectrograms or implement raw data loading.")
        return
    
    else:
        raise ValueError("Must provide either --raw_data_dir or --spectrograms_dir")
    
    # ====================================================================
    # STAGE 2: ORGANIZE DATA
    # ====================================================================
    
    print("\n" + "="*80)
    print("STAGE 2: ORGANIZING DATA INTO TRAIN/VAL SPLITS")
    print("="*80)
    
    split_dir = output_dir / 'splits'
    split_info = organize_data_for_training(
        str(processed_dir),
        str(split_dir),
        train_ratio=0.8,
        seed=42
    )
    
    # ====================================================================
    # STAGE 3: RUN PRE-TRAINING
    # ====================================================================
    
    print("\n" + "="*80)
    print("STAGE 3: RUNNING PRE-TRAINING")
    print("="*80)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        'python', 'pretrain_vit.py',
        '--data_dir', str(split_info['train_dir']),
        '--val_data_dir', str(split_info['val_dir']),
        '--output_dir', str(checkpoint_dir),
        '--model', model,
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--mask_ratio', str(mask_ratio),
        '--device', device,
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("PRE-TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
    except subprocess.CalledProcessError as e:
        print(f"\nError during pre-training: {e}")
        return
    
    # ====================================================================
    # STAGE 4: PREPARE OUTPUT
    # ====================================================================
    
    print("\n" + "="*80)
    print("STAGE 4: PREPARING OUTPUT")
    print("="*80)
    
    # Copy final model to output
    import shutil
    final_model_src = checkpoint_dir / 'pretrain_final.pth'
    final_model_dst = output_dir / 'pretrained_model.pth'
    
    if final_model_src.exists():
        shutil.copy(final_model_src, final_model_dst)
        print(f"Pre-trained model saved: {final_model_dst}")
    
    # Save configuration
    config = {
        'model': model,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'mask_ratio': mask_ratio,
        'device': device,
        'split_info': split_info,
        'checkpoint_dir': str(checkpoint_dir),
        'final_model': str(final_model_dst),
    }
    
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved: {config_file}")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"""
Output Directory: {output_dir}

Files Generated:
  - pretrained_model.pth: Pre-trained encoder (use for fine-tuning)
  - checkpoints/: All training checkpoints
  - splits/train/: Training data
  - splits/val/: Validation data
  - config.json: Configuration used

Next Steps:
  1. Use pretrained_model.pth in your fine-tuning script
  2. Load with: torch.load('{final_model_dst}')
  3. Freeze backbone and train regression head

Fine-tuning Example:
  
  import torch
  import vision_transformer as vits
  
  model = vits.vit_small(in_chans=128, num_classes=1)
  state = torch.load('{final_model_dst}')
  model.load_state_dict(state, strict=False)
  
  # Freeze backbone
  for p in model.patch_embed.parameters():
      p.requires_grad = False
  for p in model.blocks.parameters():
      p.requires_grad = False
  
  # Train on downstream task...
    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        'Complete EEG ViT Pre-training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Pre-train with pre-computed spectrograms
  python pretrain_pipeline.py \\
      --spectrograms_dir ./data/spectrograms/ \\
      --output_dir ./pretrained_models/ \\
      --epochs 100 \\
      --batch_size 16

  # Pre-train with raw EEG (if implemented)
  python pretrain_pipeline.py \\
      --raw_data_dir ./raw_eeg/ \\
      --output_dir ./pretrained_models/
        """
    )
    
    # Data
    parser.add_argument('--raw_data_dir', type=str, default=None,
                       help='Directory with raw EEG .mat files (optional)')
    parser.add_argument('--spectrograms_dir', type=str, default=None,
                       help='Directory with pre-computed .npz spectrograms')
    parser.add_argument('--output_dir', type=str, default='./pretrained_models/',
                       help='Output directory for models and data')
    
    # Model
    parser.add_argument('--model', type=str, default='vit_small',
                       choices=['vit_tiny', 'vit_small', 'vit_base'],
                       help='Model architecture')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Masking ratio for MAE')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.spectrograms_dir and not args.raw_data_dir:
        print("Error: Must provide either --spectrograms_dir or --raw_data_dir")
        parser.print_help()
        sys.exit(1)
    
    # Run pipeline
    run_full_pipeline(**vars(args))


if __name__ == '__main__':
    main()
