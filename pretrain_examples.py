"""
Example usage scripts for pre-training ViT on EEG spectrograms.
Shows different scenarios and configurations.
"""

import os
import sys
from pathlib import Path


# ============================================================================
# EXAMPLE 1: Basic Pre-training with Default Settings
# ============================================================================

def example_basic_pretrain():
    """
    Minimal example: Pre-train on spectrograms in a directory.
    
    Expected directory structure:
        data/spectrograms/
        ├── participant_001.npz
        ├── participant_002.npz
        └── ...
    
    Each .npz file should contain:
        - 'spectrograms': array of shape (128, n_windows, 224, 224)
    """
    
    command = """
    python pretrain_vit.py \\
        --data_dir "./data/spectrograms/" \\
        --output_dir "./pretrain_checkpoints/" \\
        --model vit_small \\
        --batch_size 16 \\
        --epochs 100 \\
        --lr 1e-4
    """
    
    print("EXAMPLE 1: Basic Pre-training")
    print("="*80)
    print(command)
    return command


# ============================================================================
# EXAMPLE 2: Pre-training with Validation Split
# ============================================================================

def example_pretrain_with_validation():
    """
    Pre-train with separate validation set.
    
    Expected directory structure:
        data/train/
        ├── participant_001.npz
        └── ...
        
        data/val/
        ├── participant_100.npz
        └── ...
    """
    
    command = """
    python pretrain_vit.py \\
        --data_dir "./data/train/" \\
        --val_data_dir "./data/val/" \\
        --output_dir "./pretrain_checkpoints/" \\
        --model vit_small \\
        --batch_size 16 \\
        --epochs 100 \\
        --lr 1e-4 \\
        --warmup_epochs 5
    """
    
    print("\nEXAMPLE 2: Pre-training with Validation")
    print("="*80)
    print(command)
    return command


# ============================================================================
# EXAMPLE 3: Resume Training from Checkpoint
# ============================================================================

def example_resume_training():
    """
    Resume pre-training from a saved checkpoint.
    """
    
    command = """
    python pretrain_vit.py \\
        --data_dir "./data/spectrograms/" \\
        --val_data_dir "./data/val/" \\
        --output_dir "./pretrain_checkpoints/" \\
        --model vit_small \\
        --batch_size 16 \\
        --epochs 150 \\
        --lr 1e-4 \\
        --resume "./pretrain_checkpoints/pretrain_ckpt_ep0050.pth"
    """
    
    print("\nEXAMPLE 3: Resume from Checkpoint")
    print("="*80)
    print(command)
    return command


# ============================================================================
# EXAMPLE 4: Aggressive Masking for Harder Pre-training
# ============================================================================

def example_aggressive_masking():
    """
    Pre-train with aggressive masking (more challenging task).
    Default mask_ratio is 0.75 (mask 75% of patches).
    Here we use 0.9 (mask 90%).
    """
    
    command = """
    python pretrain_vit.py \\
        --data_dir "./data/spectrograms/" \\
        --val_data_dir "./data/val/" \\
        --output_dir "./pretrain_checkpoints/" \\
        --model vit_small \\
        --batch_size 16 \\
        --epochs 100 \\
        --lr 1e-4 \\
        --mask_ratio 0.9
    """
    
    print("\nEXAMPLE 4: Aggressive Masking (90%)")
    print("="*80)
    print(command)
    return command


# ============================================================================
# EXAMPLE 5: Pre-training ViT-Base (Larger Model)
# ============================================================================

def example_pretrain_vit_base():
    """
    Pre-train larger ViT-Base model (more parameters, slower training).
    """
    
    command = """
    python pretrain_vit.py \\
        --data_dir "./data/spectrograms/" \\
        --val_data_dir "./data/val/" \\
        --output_dir "./pretrain_checkpoints_base/" \\
        --model vit_base \\
        --batch_size 8 \\
        --epochs 100 \\
        --lr 5e-5 \\
        --warmup_epochs 5
    """
    
    print("\nEXAMPLE 5: Pre-training ViT-Base")
    print("="*80)
    print(command)
    return command


# ============================================================================
# EXAMPLE 6: Using Pre-trained Model for Fine-tuning
# ============================================================================

def example_finetune_from_pretrained():
    """
    After pre-training is complete, use the pre-trained encoder for fine-tuning.
    
    This shows how to load the pre-trained backbone and train just the head.
    """
    
    code = """
    import torch
    import vision_transformer as vits
    
    # Load pre-trained encoder
    pretrain_checkpoint = './pretrain_checkpoints/pretrain_final.pth'
    
    # Create model with classification head
    model = vits.vit_small(
        patch_size=16,
        img_size=[224],
        in_chans=128,
        num_classes=1  # Regression: reaction time or p-factor
    )
    
    # Load pre-trained weights
    state_dict = torch.load(pretrain_checkpoint)
    
    # Note: encoder is wrapped, so we need to load into the encoder part
    # Load into patch_embed and blocks
    for key in list(state_dict.keys()):
        if key.startswith('patch_embed') or key.startswith('blocks'):
            model.state_dict()[key].copy_(state_dict[key])
    
    # Freeze backbone
    for name, param in model.named_parameters():
        if name.startswith('patch_embed') or name.startswith('blocks'):
            param.requires_grad = False
    
    # Only the head and CLS token are trainable
    print("Pre-trained backbone frozen, only head is trainable")
    
    # Now train the model on your downstream task
    # (e.g., reaction time prediction or P-factor prediction)
    """
    
    print("\nEXAMPLE 6: Fine-tuning from Pre-trained Model")
    print("="*80)
    print(code)
    return code


# ============================================================================
# EXAMPLE 7: Full Training Script for Fine-tuning
# ============================================================================

def example_full_finetune_script():
    """
    Complete fine-tuning script that uses pre-trained model.
    """
    
    code = """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    import vision_transformer as vits
    
    # Hyperparameters
    PRETRAIN_CHECKPOINT = './pretrain_checkpoints/pretrain_final.pth'
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 5e-4
    DEVICE = 'cuda'
    
    # Load pre-trained model
    model = vits.vit_small(
        patch_size=16,
        img_size=[224],
        in_chans=128,
        num_classes=1
    )
    
    # Load pre-trained weights
    pretrain_state = torch.load(PRETRAIN_CHECKPOINT)
    
    # Copy pre-trained weights to model
    for key in list(pretrain_state.keys()):
        if key in model.state_dict():
            model.state_dict()[key].copy_(pretrain_state[key])
    
    print("Pre-trained weights loaded")
    
    # Freeze backbone (keep encoder fixed)
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    
    for param in model.blocks.parameters():
        param.requires_grad = False
    
    for param in model.norm.parameters():
        param.requires_grad = False
    
    # Only CLS token and head are trainable
    print("Backbone frozen, only head is trainable")
    
    model = model.to(DEVICE)
    
    # Optimizer (only for trainable parameters)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.05
    )
    
    # Loss function (MSE for regression)
    criterion = nn.MSELoss()
    
    # Training loop (pseudo-code)
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (spectrograms, targets) in enumerate(train_loader):
            spectrograms = spectrograms.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            outputs = model(spectrograms)
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
    """
    
    print("\nEXAMPLE 7: Complete Fine-tuning Script")
    print("="*80)
    print(code)
    return code


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + "EEG ViT Pre-training Examples".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    examples = [
        example_basic_pretrain,
        example_pretrain_with_validation,
        example_resume_training,
        example_aggressive_masking,
        example_pretrain_vit_base,
        example_finetune_from_pretrained,
        example_full_finetune_script,
    ]
    
    for example_func in examples:
        example_func()
        print()
    
    print("\n" + "="*80)
    print("QUICK START GUIDE")
    print("="*80)
    print("""
1. Prepare your data:
   - Organize .npz files in a directory
   - Each .npz should have 'spectrograms' key: shape (128, n_windows, 224, 224)

2. Run pre-training:
   python pretrain_vit.py \\
       --data_dir ./data/spectrograms/ \\
       --output_dir ./pretrain_checkpoints/ \\
       --batch_size 16 \\
       --epochs 100

3. Monitor training:
   - Check logs for loss values
   - Checkpoints saved every --save_interval epochs

4. Use pre-trained model:
   - Load pretrain_final.pth in your fine-tuning script
   - Freeze backbone, train only regression head

Key Parameters:
   --mask_ratio: What fraction of patches to mask (0.75 = 75%)
   --warmup_epochs: How many epochs to warm up learning rate
   --weight_decay: L2 regularization strength
   --patch_size: Size of patches (16x16 standard)

For detailed info, see README_PRETRAIN.md
    """)
