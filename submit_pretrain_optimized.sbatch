#!/bin/bash
# submit_pretrain_optimized.sbatch
# Optimized pretraining script with cached dataset index and larger batch size
#
# This script uses:
#   1. Cached dataset index (eliminates 562s indexing delay)
#   2. Larger batch size for better GPU utilization
#   3. Same learning rate (scaled for batch size)

#SBATCH --job-name=eeg_mae_opt
#SBATCH --account=anwargrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/mae_opt_%j.out
#SBATCH --error=logs/mae_opt_%j.err
set -euo pipefail

PROJECT=~/EEG_challenge
cd "$PROJECT"
mkdir -p logs

DATA_DIR="/scratch/hakati/spectrograms_npy"
OUT_DIR="$PROJECT/pretrain_checkpoints_full"
INDEX_CACHE="$PROJECT/dataset_index.pkl"

mkdir -p "$OUT_DIR"

set +u; source "$PROJECT/env/activate_env.sh"; set -u

echo "Node        : $(hostname)"
echo "GPU         : ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda_avail', torch.cuda.is_available())"

# Build cached index if it doesn't exist
if [[ ! -f "$INDEX_CACHE" ]]; then
    echo "Building dataset index cache (one-time)..."
    python cache_dataset_index.py --data_dir "$DATA_DIR" --cache_path "$INDEX_CACHE" || exit 1
    echo "✓ Index cached successfully"
fi

RESUME_ARG=""
LATEST=$(ls -1t "$OUT_DIR"/pretrain_ckpt_ep*.pth 2>/dev/null | head -n1 || true)
if [[ -n "${LATEST}" ]]; then
    echo "Resuming from: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
else
    echo "No checkpoint found -> starting from scratch."
fi

# Run with optimized batch size and cached index
srun python pretrain_mae.py \
    --data_dir     "$DATA_DIR" \
    --output_dir   "$OUT_DIR" \
    --index_cache  "$INDEX_CACHE" \
    --npz_key      spectrograms \
    --model        vit_small \
    --epochs       30 \
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
