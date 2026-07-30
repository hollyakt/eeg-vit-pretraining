"""
QUICK START: EEG ViT Pre-training

This file explains how to get started with pre-training immediately.
"""

# ============================================================================
# FILES CREATED FOR PRE-TRAINING
# ============================================================================

FILES_CREATED = {
    'pretrain_vit.py': {
        'description': 'Main pre-training script - Masked Autoencoder (MAE)',
        'purpose': 'Self-supervised learning on EEG spectrograms',
        'command': 'python pretrain_vit.py --data_dir ./data/ --output_dir ./checkpoints/',
    },
    'pretrain_pipeline.py': {
        'description': 'End-to-end pipeline - data prep to trained model',
        'purpose': 'Orchestrates entire workflow with automatic data organization',
        'command': 'python pretrain_pipeline.py --spectrograms_dir ./data/ --output_dir ./models/',
    },
    'pretrain_examples.py': {
        'description': 'Example commands and usage patterns',
        'purpose': 'Reference for different training scenarios',
        'command': 'python pretrain_examples.py',
    },
    'utils.py': {
        'description': 'Utility functions (weight init, seed fixing)',
        'purpose': 'Helper functions for training',
        'usage': 'Imported by pretrain_vit.py',
    },
    'README_PRETRAIN.md': {
        'description': 'Comprehensive documentation',
        'purpose': 'Detailed guide for all aspects of pre-training',
        'link': 'See README_PRETRAIN.md for full details',
    },
}


# ============================================================================
# 5-MINUTE QUICK START
# ============================================================================

QUICK_START = """
STEP 1: Prepare Your Data
═══════════════════════════════════════════════════════════════════════════

Expected format: .npz files with 128-channel spectrograms

Create a directory with your .npz files:
  data/spectrograms/
  ├── participant_001.npz     (shape: 128 × n_windows × 224 × 224)
  ├── participant_002.npz
  └── participant_003.npz


STEP 2: Run Pre-training (5 Epochs Test)
═══════════════════════════════════════════════════════════════════════════

cd /Users/holly/Desktop/EEG Challenge Code

# Quick test run
python pretrain_vit.py \\
    --data_dir ./data/spectrograms/ \\
    --output_dir ./pretrain_checkpoints/ \\
    --epochs 5 \\
    --batch_size 16

Expected output:
  - Loaded 1500+ spectrogram samples
  - Model created (22M parameters)
  - Training begins...
  - Loss decreases over epochs


STEP 3: Run Full Pre-training (100 Epochs)
═══════════════════════════════════════════════════════════════════════════

# Full training with validation
python pretrain_vit.py \\
    --data_dir ./data/train/ \\
    --val_data_dir ./data/val/ \\
    --output_dir ./pretrain_checkpoints/ \\
    --epochs 100 \\
    --batch_size 16 \\
    --lr 1e-4 \\
    --warmup_epochs 5

This will:
  1. Load training and validation spectrograms
  2. Initialize ViT-Small model
  3. Train with Masked Autoencoder (MAE) loss
  4. Save checkpoints every 10 epochs
  5. Create pretrain_final.pth (use for fine-tuning)


STEP 4: Use Pre-trained Model for Fine-tuning
═══════════════════════════════════════════════════════════════════════════

# In your fine-tuning script:

import torch
import vision_transformer as vits

# Create model with regression head (for RT or P-factor prediction)
model = vits.vit_small(
    patch_size=16,
    img_size=[224],
    in_chans=128,
    num_classes=1  # Regression target
)

# Load pre-trained encoder
pretrain_path = './pretrain_checkpoints/pretrain_final.pth'
state_dict = torch.load(pretrain_path)
model.load_state_dict(state_dict, strict=False)

# Freeze backbone (keep it fixed)
for param in model.patch_embed.parameters():
    param.requires_grad = False
for param in model.blocks.parameters():
    param.requires_grad = False

# Only the head is trained
print("Backbone frozen ✓")
print("Ready for fine-tuning")

# Now train on your downstream task...
"""


# ============================================================================
# KEY PARAMETERS EXPLAINED
# ============================================================================

PARAMETERS_GUIDE = """
IMPORTANT PARAMETERS
═══════════════════════════════════════════════════════════════════════════

--data_dir
  Location of training .npz files
  REQUIRED: Must contain spectrograms

--val_data_dir
  Location of validation .npz files
  OPTIONAL: If not provided, no validation

--output_dir
  Where to save checkpoints
  DEFAULT: ./pretrain_checkpoints/

--model
  Model architecture: vit_tiny, vit_small, vit_base
  DEFAULT: vit_small (recommended for most cases)

--epochs
  Number of training epochs
  TYPICAL: 100 for full training, 5 for testing

--batch_size
  Batch size (samples per iteration)
  TYPICAL: 16-32 (depends on GPU memory)
  MEMORY: 16→8GB, 32→16GB, 64→24GB

--lr
  Learning rate
  DEFAULT: 1e-4 (good for most cases)
  TOO HIGH: Training unstable
  TOO LOW: Training very slow

--mask_ratio
  Fraction of patches to mask (0.75 = 75%)
  DEFAULT: 0.75 (standard)
  AGGRESSIVE: 0.9 (harder pre-training)

--warmup_epochs
  Number of epochs to warm up learning rate
  DEFAULT: 5 (recommended)

--device
  cuda (GPU) or cpu
  DEFAULT: cuda
  GPU RECOMMENDED: ~10x faster


CHOOSING BATCH SIZE
═══════════════════════════════════════════════════════════════════════════

GPU Memory vs Batch Size:
  - 8 GB GPU:   batch_size = 8-16
  - 12 GB GPU:  batch_size = 16-32
  - 16 GB GPU:  batch_size = 32-48
  - 24 GB GPU:  batch_size = 64+

Not enough memory? → Reduce batch_size
Too slow? → Increase batch_size


CHOOSING NUMBER OF EPOCHS
═══════════════════════════════════════════════════════════════════════════

Data Size vs Epochs:
  - < 1,000 samples:  50-100 epochs (small dataset)
  - 1,000-10,000:    100-200 epochs (medium)
  - > 10,000:        50-100 epochs (large)

Typical training time:
  - 100 epochs: ~10 hours on V100 GPU
  - 100 epochs: ~30 hours on RTX 3090
  - 100 epochs: ~50 hours on CPU (not recommended)
"""


# ============================================================================
# EXPECTED OUTPUT
# ============================================================================

EXPECTED_OUTPUT = """
WHAT YOU SHOULD SEE
═══════════════════════════════════════════════════════════════════════════

Initial output:
  ================================================================================
  EEG ViT Pre-training
  ================================================================================
  Timestamp: 2024-06-09 10:00:00
  Device: cuda
  Data directory: ./data/spectrograms/
  Model: vit_small
  Batch size: 16
  Learning rate: 0.0001
  Mask ratio: 0.75
  ================================================================================

Loading data:
  Found 150 .npz files in ./data/spectrograms/
  Loaded 1500 spectrogram samples

Model creation:
  Building model: vit_small
  Model created. Total parameters: 22,050,000

Training progress:
  Epoch 0 [100/94] Loss: 0.234567 (Avg: 0.245678) LR: 4.25e-06
  Epoch 0 [200/94] Loss: 0.210123 (Avg: 0.231456) LR: 8.50e-06
  Epoch 0: Train Loss = 0.187654
  Epoch 0: Val Loss = 0.192345
  Checkpoint saved: ./pretrain_checkpoints/pretrain_ckpt_ep0000.pth

Loss should DECREASE over epochs!
  ✓ Epoch 0: Loss = 0.250
  ✓ Epoch 1: Loss = 0.220
  ✓ Epoch 2: Loss = 0.195
  ✓ Epoch 10: Loss = 0.150


WHAT MIGHT GO WRONG
═══════════════════════════════════════════════════════════════════════════

❌ "No .npz files found"
   → Check data_dir exists and contains .npz files
   → Verify file names match pattern

❌ "CUDA out of memory"
   → Reduce batch_size (--batch_size 8)
   → Use --device cpu (much slower)
   → Reduce image size in preprocessing

❌ "Loss not decreasing"
   → Increase learning rate (--lr 1e-3)
   → Reduce number of warmup epochs
   → Check data quality

❌ "FileNotFoundError: pretrain_vit.py"
   → Make sure you're in correct directory
   → cd /Users/holly/Desktop/EEG Challenge Code

✓ Training is working correctly if:
  - Loss decreases smoothly
  - No CUDA errors
  - Checkpoints are being saved
  - Memory usage is stable
"""


# ============================================================================
# COMMAND REFERENCE
# ============================================================================

COMMAND_REFERENCE = """
QUICK COMMAND REFERENCE
═══════════════════════════════════════════════════════════════════════════

# Test on 5 epochs (quick validation)
python pretrain_vit.py --data_dir ./data/ --epochs 5

# Full pre-training (100 epochs)
python pretrain_vit.py \\
    --data_dir ./data/train/ \\
    --val_data_dir ./data/val/ \\
    --epochs 100 --batch_size 16

# Aggressive masking (90%)
python pretrain_vit.py \\
    --data_dir ./data/ \\
    --epochs 100 \\
    --mask_ratio 0.9

# Larger model (ViT-Base)
python pretrain_vit.py \\
    --data_dir ./data/ \\
    --model vit_base \\
    --batch_size 8 \\
    --epochs 100

# Resume from checkpoint
python pretrain_vit.py \\
    --data_dir ./data/ \\
    --resume ./pretrain_checkpoints/pretrain_ckpt_ep0050.pth \\
    --epochs 150

# See all options
python pretrain_vit.py --help

# Run example commands
python pretrain_examples.py
"""


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

TROUBLESHOOTING = """
TROUBLESHOOTING GUIDE
═══════════════════════════════════════════════════════════════════════════

Problem: "ModuleNotFoundError: No module named 'vision_transformer'"
Solution:
  - Make sure vision_transformer.py is in the same directory
  - Check that the file isn't corrupted
  - Verify Python can see the file: python -c "import vision_transformer"

Problem: "CUDA out of memory"
Solution:
  - Reduce batch size: --batch_size 8
  - Clear GPU memory: torch.cuda.empty_cache()
  - Check GPU: nvidia-smi
  - Use CPU instead: --device cpu (slower but works)

Problem: "Loss not decreasing" or "Loss = NaN"
Solution:
  - Check data is properly normalized [0, 1]
  - Reduce learning rate: --lr 1e-5
  - Check for NaN in data: np.isnan(data).any()
  - Try gradient clipping (already enabled in code)

Problem: "FileNotFoundError: No .npz files found"
Solution:
  - Verify directory path is correct
  - Check .npz files exist: ls -la /path/to/data/
  - Verify file format with: import numpy as np; np.load('file.npz').files
  - Should show 'spectrograms' as a key

Problem: "Out of memory" (RAM, not CUDA)
Solution:
  - Reduce number of workers: --num_workers 0
  - Process data in smaller batches
  - Check system RAM: free -h
  - Process one participant at a time

Problem: "Slow training"
Solution:
  - Increase batch_size (if memory allows)
  - Increase num_workers: --num_workers 8
  - Use SSD instead of HDD for data
  - Use GPU: --device cuda
  - Profile with: python -m cProfile pretrain_vit.py ...


DATA QUALITY CHECKS
═══════════════════════════════════════════════════════════════════════════

Run before pre-training:

import numpy as np
from pathlib import Path

data_dir = Path('./data/spectrograms/')

for npz_file in data_dir.glob('*.npz'):
    data = np.load(npz_file)
    specs = data['spectrograms']
    
    print(f"{npz_file.name}:")
    print(f"  Shape: {specs.shape}")
    print(f"  dtype: {specs.dtype}")
    print(f"  Min: {specs.min():.4f}, Max: {specs.max():.4f}")
    print(f"  Has NaN: {np.isnan(specs).any()}")
    print(f"  Has Inf: {np.isinf(specs).any()}")
    
    # Expected: shape (128, n_windows, 224, 224), values in [0, 1]
    if specs.shape[0] != 128:
        print("  WARNING: Not 128 channels!")
    if specs.min() < -0.01 or specs.max() > 1.01:
        print("  WARNING: Values not in [0, 1]!")

print("\\n✓ Data quality check complete")
"""


# ============================================================================
# NEXT STEPS
# ============================================================================

NEXT_STEPS = """
NEXT STEPS AFTER PRE-TRAINING
═══════════════════════════════════════════════════════════════════════════

1. CHECK TRAINING RESULTS
   - Look at checkpoints/ directory
   - Monitor loss curve (should decrease)
   - Validate loss should be close to training loss

2. SAVE PRE-TRAINED MODEL
   - Copy pretrain_final.pth to safe location
   - This is your pre-trained backbone

3. PREPARE FINE-TUNING
   - Gather labeled data for your task (RT or P-factor)
   - Split into train/val (80/20)
   - Create data loaders

4. FINE-TUNE MODEL
   - Load pre-trained weights
   - Freeze backbone layers
   - Train only regression head
   - Use eval_finetune_model1.py or eval_finetune_model2.py as reference

5. EVALUATE
   - Use test set to evaluate performance
   - Compare to baseline (training without pre-training)
   - Report metrics

EXPECTED IMPROVEMENTS
  - Pre-training typically improves downstream performance by 5-15%
  - Faster convergence during fine-tuning
  - Better generalization to new data


FILE ORGANIZATION AFTER PRE-TRAINING
═══════════════════════════════════════════════════════════════════════════

pretrain_checkpoints/
├── pretrain_ckpt_ep0000.pth    (checkpoint from epoch 0)
├── pretrain_ckpt_ep0010.pth    (checkpoint from epoch 10)
├── pretrain_ckpt_ep0020.pth    (checkpoint from epoch 20)
├── ...
└── pretrain_final.pth           ← USE THIS FOR FINE-TUNING

Copy pretrain_final.pth to your fine-tuning project.
"""


# ============================================================================
# MAIN
# ============================================================================

def print_section(title: str, content: str):
    """Print a formatted section."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + title.center(78) + "║")
    print("╚" + "="*78 + "╝\n")
    print(content)


if __name__ == '__main__':
    import sys
    
    print_section("QUICK START: EEG ViT PRE-TRAINING", "")
    
    print("FILES CREATED FOR PRE-TRAINING")
    print("─" * 80)
    for fname, info in FILES_CREATED.items():
        print(f"\n{fname}")
        print(f"  {info['description']}")
        print(f"  Purpose: {info['purpose']}")
    
    print_section("GET STARTED IN 5 MINUTES", QUICK_START)
    
    print_section("KEY PARAMETERS", PARAMETERS_GUIDE)
    
    print_section("EXPECTED OUTPUT", EXPECTED_OUTPUT)
    
    print_section("COMMAND REFERENCE", COMMAND_REFERENCE)
    
    print_section("TROUBLESHOOTING", TROUBLESHOOTING)
    
    print_section("NEXT STEPS", NEXT_STEPS)
    
    print("\n" + "="*80)
    print("Ready to start? Run this command:")
    print("="*80)
    print("""
python pretrain_vit.py \\
    --data_dir ./data/spectrograms/ \\
    --output_dir ./pretrain_checkpoints/ \\
    --epochs 100 \\
    --batch_size 16
    """)
    
    print("=" * 80)
    print("For more details, see README_PRETRAIN.md")
    print("=" * 80 + "\n")
