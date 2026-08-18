#!/bin/bash
#SBATCH --job-name=eeg_mae_dbg
#SBATCH --account=anwargrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/mae_dbg_%j.out
#SBATCH --error=logs/mae_dbg_%j.err
set -euo pipefail

PROJECT=~/EEG_challenge
cd "$PROJECT"
mkdir -p logs

DATA_DIR="/scratch/hakati/spectrograms"
OUT_DIR="$PROJECT/pretrain_checkpoints_debug"
mkdir -p "$OUT_DIR"

set +u; source "$PROJECT/env/activate_env.sh"; set -u

echo "Node        : $(hostname)"
echo "GPU         : ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda_avail', torch.cuda.is_available())"

srun python pretrain_mae.py \
    --data_dir     "$DATA_DIR" \
    --output_dir   "$OUT_DIR" \
    --npz_key      spectrograms \
    --model        vit_small \
    --epochs       1 \
    --batch_size   16 \
    --lr           1.5e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 1 \
    --mask_ratio   0.75 \
    --num_workers 0 \
    --device       cuda \
    --log_interval 10 \
    --save_interval 1

echo "Done."
