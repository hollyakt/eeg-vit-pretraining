#!/bin/bash
#SBATCH --job-name=eval_after_finetune
#SBATCH --account=anwargrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval_after_finetune_%j.out
#SBATCH --error=logs/eval_after_finetune_%j.err
#SBATCH --dependency=afterok:493160  # Update this to match the actual fine-tune job ID

set -euo pipefail

PROJECT=~/EEG_challenge
cd "$PROJECT"
mkdir -p logs

set +u; source "$PROJECT/env/activate_env.sh"; set -u

echo "Node: $(hostname)"
echo "Using python: $(which python)"

python evaluate_after_finetune.py \
  --checkpoint pretrain_finetune_results/checkpoint_latest.pth \
  --data_location /scratch/hakati/spectrograms \
  --val_fold 1 \
  --batch_size 8 \
  --device cuda \
  --num_workers 4
