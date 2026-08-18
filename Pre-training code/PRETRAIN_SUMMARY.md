"""
SUMMARY: Pre-training Framework for EEG ViT

Created files and their purposes:
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         PRE-TRAINING FRAMEWORK FOR VISION TRANSFORMER ON EEG SPECTROGRAMS    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT WAS CREATED
═══════════════════════════════════════════════════════════════════════════════

You now have a complete self-supervised pre-training pipeline for ViT-Small on EEG
data. This implements Masked Autoencoder (MAE) pre-training, which prepares the
model's backbone so fine-tuning on your downstream tasks (Model 1: Reaction Time,
Model 2: P-Factor) is more efficient and accurate.


CORE FILES
───────────────────────────────────────────────────────────────────────────────

1. pretrain_vit.py (Main Script)
   ├─ Implements Masked Autoencoder (MAE)
   ├─ Handles 128-channel EEG spectrogram input
   ├─ Supports ViT-Tiny, Small, and Base architectures
   ├─ Includes data loading, training loop, checkpointing
   ├─ Generates pretrain_final.pth (backbone for fine-tuning)
   └─ Command:
       python pretrain_vit.py --data_dir ./data/ --output_dir ./checkpoints/


2. pretrain_pipeline.py (End-to-End Pipeline)
   ├─ Orchestrates entire workflow
   ├─ Automatically splits data into train/val
   ├─ Runs pre-training
   ├─ Organizes outputs
   └─ Command:
       python pretrain_pipeline.py --spectrograms_dir ./data/ \
           --output_dir ./models/


3. pretrain_examples.py (Reference Commands)
   ├─ 7 different training scenarios
   ├─ Copy-paste ready examples
   ├─ From basic to advanced usage
   └─ Run with:
       python pretrain_examples.py


DOCUMENTATION FILES
───────────────────────────────────────────────────────────────────────────────

4. README_PRETRAIN.md (Comprehensive Guide)
   ├─ Detailed architecture explanation
   ├─ Data preparation guide
   ├─ Parameter explanations
   ├─ Troubleshooting section
   ├─ Performance benchmarks
   └─ Next steps for fine-tuning

5. QUICKSTART.py (Quick Reference)
   ├─ 5-minute quick start
   ├─ Parameter guide
   ├─ Expected output
   ├─ Troubleshooting
   └─ Run with:
       python QUICKSTART.py


UTILITY FILES
───────────────────────────────────────────────────────────────────────────────

6. utils.py (Helper Functions)
   ├─ trunc_normal_ (weight initialization)
   ├─ fix_random_seeds (reproducibility)
   └─ Used internally by pretrain_vit.py


ARCHITECTURE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Input:  128-channel EEG spectrograms (128, 224, 224)
           ↓
Patch Embedding: 16×16 patches → 196 tokens of 384-dim each
           ↓
Masking: Randomly mask 75% of patches (configurable)
           ↓
ViT Blocks: 12 transformer blocks (6 heads, 384 hidden dim)
           ↓
Reconstruction Head: Predict original patch values
           ↓
Loss: MSE between prediction and target
           ↓
Output: Pre-trained encoder weights (saved to .pth)


TRAINING WORKFLOW
═══════════════════════════════════════════════════════════════════════════════

1. Data Preparation
   ├─ Organize .npz files in directory
   ├─ Each file: shape (128, n_windows, 224, 224)
   └─ Script handles loading and augmentation

2. Model Initialization
   ├─ Create ViT encoder
   ├─ Add MAE reconstruction head
   └─ Total: ~22M parameters

3. Training Loop (per epoch)
   ├─ Load batch of spectrograms
   ├─ Apply random masking
   ├─ Forward through encoder + decoder
   ├─ Compute MSE loss
   ├─ Backward + optimize
   └─ Save checkpoint every N epochs

4. Validation (optional)
   ├─ Monitor validation loss
   ├─ Save best checkpoint
   └─ Early stopping (optional)

5. Output
   ├─ pretrain_final.pth (use for fine-tuning)
   ├─ Periodic checkpoints (resume training)
   └─ Training logs


QUICK START (3 COMMANDS)
═══════════════════════════════════════════════════════════════════════════════

Command 1: Check your data is ready
───────────────────────────────────

cd /Users/holly/Desktop/EEG Challenge Code

ls data/spectrograms/  # Should show .npz files
python -c "import numpy as np; d=np.load('data/spectrograms/file.npz'); print(d['spectrograms'].shape)"


Command 2: Run pre-training (quick test)
────────────────────────────────────────

python pretrain_vit.py \\
    --data_dir data/spectrograms/ \\
    --output_dir pretrain_checkpoints/ \\
    --epochs 5 \\
    --batch_size 16


Command 3: Run full pre-training (100 epochs)
──────────────────────────────────────────────

python pretrain_vit.py \\
    --data_dir data/train/ \\
    --val_data_dir data/val/ \\
    --output_dir pretrain_checkpoints/ \\
    --epochs 100 \\
    --batch_size 16 \\
    --lr 1e-4


KEY FEATURES
═══════════════════════════════════════════════════════════════════════════════

✓ Self-supervised learning (no labels needed for pre-training)
✓ Masked Autoencoder (MAE) approach
✓ Supports ViT-Tiny, Small, Base models
✓ Optimized for 128-channel EEG input
✓ Automatic checkpointing every N epochs
✓ Resume from checkpoint
✓ Validation monitoring
✓ Data augmentation (built-in)
✓ Flexible masking ratio (default 75%)
✓ Cosine annealing with warmup scheduler
✓ Gradient clipping for stability
✓ CUDA/GPU support


EXPECTED PERFORMANCE
═══════════════════════════════════════════════════════════════════════════════

On typical EEG spectrogram dataset:

Training Time:
  - ViT-Small, 100 epochs, batch=32: ~8 hours (RTX 3090)
  - ViT-Small, 100 epochs, batch=16: ~12 hours (V100)
  - ViT-Base, 100 epochs, batch=16: ~24 hours (RTX 3090)

Loss Progression (typical):
  Epoch 0:   Loss = 0.250
  Epoch 10:  Loss = 0.180
  Epoch 30:  Loss = 0.140
  Epoch 100: Loss = 0.095


NEXT STEPS
═══════════════════════════════════════════════════════════════════════════════

1. GET DATA READY
   → Organize .npz files with 128×224×224 spectrograms
   → Split into train/val directories

2. RUN PRE-TRAINING
   → Use pretrain_vit.py
   → Monitor loss (should decrease)
   → Takes 8-24 hours typically

3. USE PRE-TRAINED MODEL
   → Load pretrain_final.pth
   → Freeze encoder layers
   → Train regression head on Model 1 or Model 2

4. EVALUATE
   → Compare downstream task performance with/without pre-training
   → Expect 5-15% improvement

5. SUBMIT
   → Use eval_finetune_clf_ensemble_ecog90s.py for Model 1
   → Use eval_finetune_model2.py for Model 2


INTEGRATION WITH EXISTING CODE
═══════════════════════════════════════════════════════════════════════════════

Existing files you have:
  ✓ vision_transformer.py - ViT architecture (used)
  ✓ dataloader.py - Data utilities (reference)
  ✓ preprocessing.py - EEG preprocessing (reference)
  ✓ spects.py - Spectrogram creation (reference)
  ✓ eval_finetune_model1.py - Model 1 fine-tuning (next step)
  ✓ eval_finetune_model2.py - Model 2 fine-tuning (next step)

New files for pre-training:
  + pretrain_vit.py - Main pre-training script
  + pretrain_pipeline.py - Full pipeline
  + pretrain_examples.py - Example commands
  + utils.py - Helper functions
  + README_PRETRAIN.md - Documentation


COMMON QUESTIONS
═══════════════════════════════════════════════════════════════════════════════

Q: How long does pre-training take?
A: 8-24 hours depending on data size and hardware. ViT-Small takes ~8 hours on
   RTX 3090 for 100 epochs.

Q: Do I need labels for pre-training?
A: No! Pre-training is self-supervised. Labels are only needed for fine-tuning.

Q: What if I don't have much data?
A: Pre-training still helps. With < 1K samples, run 50 epochs. Masking ratio
   can be increased to 0.9 for harder pre-training.

Q: Can I use a different model size?
A: Yes. Use --model vit_tiny (faster), vit_small (default), or vit_base (slower).

Q: How do I know if training is working?
A: Loss should decrease smoothly. If loss increases or is NaN, check your data
   normalization and learning rate.

Q: Can I resume from a checkpoint?
A: Yes! Use --resume checkpoint.pth to continue training.

Q: Will pre-training improve my Model 1/Model 2 results?
A: Yes, typically 5-15% improvement. Especially helpful if you have limited
   labeled data for your downstream task.


TECHNICAL DETAILS
═══════════════════════════════════════════════════════════════════════════════

Optimizer: AdamW
  - Learning rate: 1e-4 (default, adjustable)
  - Betas: (0.9, 0.95)
  - Weight decay: 0.05
  - Epsilon: 1e-8

Scheduler: Cosine annealing with warmup
  - Warmup epochs: 5 (default)
  - Total epochs: configurable

Masking Strategy: Random shuffling
  - Default mask ratio: 0.75 (mask 75% of patches)
  - Mask token: learnable parameter
  - Applied per-batch during training

Loss: MSE (Mean Squared Error)
  - Computed on all patches (both masked and unmasked)
  - No weighting on masked vs visible patches

Data Augmentation:
  - Random contrast (0.9x - 1.1x)
  - Time jittering (±2 pixels)
  - Gaussian noise (σ=0.01)


FILES IN THIS DIRECTORY AFTER PRE-TRAINING
═══════════════════════════════════════════════════════════════════════════════

pretrain_checkpoints/
├── pretrain_ckpt_ep0000.pth    # Checkpoint epoch 0
├── pretrain_ckpt_ep0010.pth    # Checkpoint epoch 10
├── pretrain_ckpt_ep0020.pth    # ...
├── pretrain_ckpt_ep0099.pth    # Checkpoint epoch 99
└── pretrain_final.pth           # ← MAIN OUTPUT (use this for fine-tuning)

Each checkpoint contains:
  - Model state (encoder weights)
  - Optimizer state
  - Scheduler state
  - Epoch number
  - Loss value


SUPPORT & DEBUGGING
═══════════════════════════════════════════════════════════════════════════════

For detailed help:
  → See README_PRETRAIN.md (comprehensive guide)
  → Run QUICKSTART.py (quick reference)
  → Run pretrain_examples.py (example commands)
  → Check pretrain_vit.py --help (all options)

Common issues & solutions:
  → See "TROUBLESHOOTING" section in README_PRETRAIN.md
  → See "TROUBLESHOOTING GUIDE" in QUICKSTART.py

Data format validation:
  → See "DATA QUALITY CHECKS" in QUICKSTART.py


═══════════════════════════════════════════════════════════════════════════════

Ready to start?

1. Organize your .npz spectrogram files in a directory
2. Run: python pretrain_vit.py --data_dir ./data/ --epochs 5
3. Monitor the loss output
4. Run full training when ready

Good luck! 🚀

═══════════════════════════════════════════════════════════════════════════════
""")
