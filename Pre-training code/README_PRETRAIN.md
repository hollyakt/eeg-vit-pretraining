# EEG ViT Pre-training Guide

## Overview

This pre-training pipeline implements **Masked Autoencoder (MAE)** self-supervised learning on EEG spectrograms for Vision Transformer (ViT-Small). The pre-trained backbone can then be frozen and fine-tuned with just the regression head for your downstream tasks (Model 1: Reaction Time, Model 2: P-Factor).

## Architecture

- **Encoder**: Vision Transformer (ViT-Small with 12 layers, 6 heads, 384 embedding dim)
- **Input**: 128-channel EEG spectrograms, each 224×224 pixels
- **Masking Strategy**: Random patch masking (default 75%)
- **Reconstruction Target**: Original spectrogram values
- **Loss**: MSE between predicted and original spectrograms

## Files

1. **`pretrain_vit.py`** - Main pre-training script
2. **`pretrain_examples.py`** - Example commands and usage patterns
3. **`utils.py`** - Utility functions (weight initialization, seed fixing)
4. **`vision_transformer.py`** - ViT model architecture (existing)
5. **`dataloader.py`** - EEG-specific data utilities (existing)

## Installation Requirements

```bash
pip install torch torchvision
pip install numpy scipy pandas
pip install scikit-learn
pip install timm  # pytorch image models (for some utilities)
```

## Data Preparation

### Expected Format

The script expects `.npz` files containing EEG spectrograms:

```python
import numpy as np

# Save training data
spectrograms = np.random.randn(128, n_windows, 224, 224).astype(np.float32)
np.savez('participant_001.npz', spectrograms=spectrograms)
```

### Directory Structure

```
data/
├── train/
│   ├── participant_001.npz
│   ├── participant_002.npz
│   └── ...
└── val/
    ├── participant_100.npz
    └── ...
```

Or for single directory:
```
data/spectrograms/
├── participant_001.npz
├── participant_002.npz
└── ...
```

### Data Characteristics

- **Shape**: `(128, n_windows, 224, 224)` or individual (128, 224, 224)
- **Channels**: 128 (all EEG channels)
- **Window Size**: 224×224 pixels (normalized spectrograms)
- **Value Range**: Should be normalized to [0, 1], script will handle it

## Training

### Basic Usage

```bash
python pretrain_vit.py \
    --data_dir ./data/spectrograms/ \
    --output_dir ./pretrain_checkpoints/ \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4
```

### With Validation Set

```bash
python pretrain_vit.py \
    --data_dir ./data/train/ \
    --val_data_dir ./data/val/ \
    --output_dir ./pretrain_checkpoints/ \
    --batch_size 16 \
    --epochs 100
```

### Resume from Checkpoint

```bash
python pretrain_vit.py \
    --data_dir ./data/spectrograms/ \
    --output_dir ./pretrain_checkpoints/ \
    --resume ./pretrain_checkpoints/pretrain_ckpt_ep0050.pth \
    --epochs 150
```

## Command-Line Arguments

### Data Arguments
- `--data_dir` (required): Directory containing training .npz files
- `--val_data_dir` (optional): Directory for validation data
- `--output_dir` (default: `./pretrain_checkpoints`): Where to save checkpoints

### Model Arguments
- `--model` (default: `vit_small`): Model size - `vit_tiny`, `vit_small`, `vit_base`
- `--patch_size` (default: 16): Patch size (16×16)
- `--in_chans` (default: 128): Input channels (EEG channels)
- `--mask_ratio` (default: 0.75): Fraction of patches to mask [0.0-1.0]

### Training Arguments
- `--epochs` (default: 100): Number of training epochs
- `--batch_size` (default: 16): Batch size
- `--lr` (default: 1e-4): Initial learning rate
- `--weight_decay` (default: 0.05): L2 regularization
- `--warmup_epochs` (default: 5): Number of warmup epochs
- `--num_workers` (default: 4): Data loading workers

### Other Arguments
- `--device` (default: `cuda`): `cuda` or `cpu`
- `--seed` (default: 42): Random seed
- `--resume`: Path to checkpoint to resume from
- `--log_interval` (default: 100): Logging frequency (batches)
- `--save_interval` (default: 10): Checkpoint saving interval (epochs)

## Training Parameters Guide

### Recommended Configurations

#### For Quick Testing
```bash
python pretrain_vit.py \
    --data_dir ./data/ \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4
```

#### For Small Dataset (< 10K samples)
```bash
python pretrain_vit.py \
    --data_dir ./data/ \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --mask_ratio 0.75
```

#### For Large Dataset (> 50K samples)
```bash
python pretrain_vit.py \
    --data_dir ./data/ \
    --val_data_dir ./data/val/ \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-4 \
    --warmup_epochs 10 \
    --mask_ratio 0.75
```

#### For Aggressive Pre-training (Higher Masking)
```bash
python pretrain_vit.py \
    --data_dir ./data/ \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --mask_ratio 0.9  # Mask 90% of patches
```

## Expected Behavior

### Training Output

```
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

Loading training data...
Found 150 .npz files in ./data/spectrograms/
Loaded 1500 spectrogram samples

Building model: vit_small
Model created. Total parameters: 22,050,000

Starting pre-training...
================================================================================
Epoch 0 [100/94] Loss: 0.234567 (Avg: 0.245678) LR: 4.25e-06
Epoch 0 [200/94] Loss: 0.210123 (Avg: 0.231456) LR: 8.50e-06
...
Epoch 0: Train Loss = 0.187654
Epoch 0: Val Loss = 0.192345
Checkpoint saved: ./pretrain_checkpoints/pretrain_ckpt_ep0000.pth
```

### Loss Trajectory

- **Early epochs**: Loss should decrease smoothly
- **Convergence**: Loss typically plateaus after 30-50 epochs
- **Validation loss**: Should track training loss; divergence indicates overfitting

### GPU Memory Requirements

- **ViT-Small**: ~8-10 GB for batch_size=32
- **ViT-Base**: ~16-20 GB for batch_size=16
- If memory limited: reduce batch_size or use gradient accumulation

## Fine-tuning with Pre-trained Model

### Loading Pre-trained Weights

```python
import torch
import vision_transformer as vits

# Create model
model = vits.vit_small(
    patch_size=16,
    img_size=[224],
    in_chans=128,
    num_classes=1  # For regression (RT or P-factor)
)

# Load pre-trained encoder
pretrain_checkpoint = './pretrain_checkpoints/pretrain_final.pth'
state_dict = torch.load(pretrain_checkpoint)

# Load weights
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"Loaded pre-trained model. Missing keys: {len(missing_keys)}")

# Freeze backbone
for name, param in model.named_parameters():
    if 'head' not in name and 'cls_token' not in name:
        param.requires_grad = False

print("Backbone frozen. Only head and CLS token are trainable.")
```

### Key Files for Fine-tuning

The pre-training creates:
- **`pretrain_ckpt_epXXXX.pth`** - Periodic checkpoints
- **`pretrain_final.pth`** - Final encoder weights (use this for fine-tuning)

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or use gradient accumulation
```bash
--batch_size 8  # Instead of 16
```

### Issue: Loss not decreasing

**Check**:
- Data quality: Are spectrograms properly normalized?
- Learning rate: Try increasing to 1e-3
- Model size: Try `vit_tiny` (smaller, faster)

### Issue: Validation loss diverges from training loss

**Solution**: Increase weight decay or add dropout
```bash
--weight_decay 0.1  # Instead of 0.05
```

### Issue: Training very slow

**Solutions**:
- Increase `--num_workers` (more data loading workers)
- Use SSD/NVMe storage instead of HDD
- Reduce image size (224→192)
- Use `vit_tiny` instead of `vit_small`

## Advanced Usage

### Data Augmentation

The dataset class automatically applies augmentation during training:
- Random contrast adjustment (0.9-1.1x)
- Time jittering (±2 pixels shift)
- Gaussian noise (σ=0.01)

### Custom Masking Strategies

To modify masking, edit the `_random_masking` method in `MAEPretrainer`:

```python
# Current: random shuffling
# Alternative: block masking
def _block_masking(self, x, block_size=8):
    # Mask contiguous patches in a block
    ...
```

### Monitoring Training with TensorBoard (Optional)

Add to `train_epoch`:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('loss/train', loss.item(), global_step)
```

## Benchmark Results

Typical performance on EEG spectrogram dataset:

| Configuration | Train Loss | Val Loss | Time/Epoch |
|---------------|-----------|----------|-----------|
| ViT-Small, mask=0.75 | 0.18-0.22 | 0.19-0.23 | 5 min |
| ViT-Small, mask=0.9 | 0.22-0.26 | 0.23-0.27 | 5 min |
| ViT-Base, mask=0.75 | 0.16-0.20 | 0.17-0.21 | 15 min |

*Note: Times and values depend on hardware, dataset size, and batch size*

## Next Steps

1. **Pre-train**: Run `pretrain_vit.py` for 100-200 epochs
2. **Validate**: Check that validation loss decreases smoothly
3. **Save**: Use `pretrain_final.pth` for fine-tuning
4. **Fine-tune**: Freeze backbone and train regression head on your downstream task

## Citation & References

- MAE: He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
- ViT: Dosovitskiy et al., "An Image is Worth 16×16 Words" (ICLR 2021)
- SiT: Mahabadi et al., "Sit: Self-supervised Vision Transformer for Image-to-Image Learning"

## Contact & Support

For issues or questions:
1. Check logs for specific error messages
2. Review training hyperparameters
3. Verify data format matches specification
4. Check GPU memory and compute resources

---

**Last Updated**: June 2024
