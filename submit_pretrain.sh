#!/bin/bash
#SBATCH --job-name=eeg_mae
#SBATCH --account=anwargrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/mae_%j.out
#SBATCH --error=logs/mae_%j.err
set -euo pipefail

PROJECT=~/EEG_challenge
cd "$PROJECT"
mkdir -p logs

DATA_DIR="/scratch/hakati/spectrograms_npy"                 # full R1-R11 spectrograms
OUT_DIR="$PROJECT/pretrain_checkpoints_full"     # fresh dir; do NOT reuse the buggy run
INDEX_CACHE="$PROJECT/dataset_index_cache.pkl"   # cached dataset index (fast startup)
mkdir -p "$OUT_DIR"

set +u; source "$PROJECT/env/activate_env.sh"; set -u

echo "Node        : $(hostname)"
echo "GPU         : ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda_avail', torch.cuda.is_available())"

# Build index cache if it doesn't exist (one-time ~10s, saves 9+ min per job)
if [[ ! -f "$INDEX_CACHE" ]]; then
    echo "Building dataset index cache (one-time, ~1 min for 25k files)..."
    python build_dataset_index.py \
        --data_dir "$DATA_DIR" \
        --output_file "$INDEX_CACHE" \
        --num_workers 8
    echo "✓ Index cache built"
fi

RESUME_ARG=""
LATEST=$(ls -1t "$OUT_DIR"/pretrain_ckpt_ep*.pth 2>/dev/null | head -n1 || true)
if [[ -n "${LATEST}" ]]; then
    echo "Resuming from: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
else
    echo "No checkpoint found -> starting from scratch."
fi

# OPTIMIZATION: Increased batch_size from 128 to 256 to better utilize H100 GPU memory
# This should increase throughput by ~20-40% without reducing model quality
# If OOM errors occur, reduce batch_size back to 128 or 192
srun python pretrain_mae.py \
    --data_dir     "$DATA_DIR" \
    --output_dir   "$OUT_DIR" \
    --index_cache  "$INDEX_CACHE" \
    --npz_key      spectrograms \
    --model        vit_small \
    --epochs       50 \
    --batch_size   256 \
    --lr           1.5e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --mask_ratio   0.75 \
    --num_workers 8 \
    --device       cuda \
    --log_interval 50 \
    --save_interval 1 \
    ${RESUME_ARG}

echo "Done."
