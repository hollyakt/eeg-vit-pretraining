#!/bin/bash
#SBATCH --job-name=finetune_full
#SBATCH --account=anwargrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

set -euo pipefail

PROJECT=~/EEG_challenge
cd "$PROJECT"
mkdir -p logs

set +u; source "$PROJECT/env/activate_env.sh"; set -u

echo "Node: $(hostname)"
echo "Using python: $(which python)"

# Adjust args as needed
python eval_finetune_model2.py \
  --finetune pretrain_checkpoints_debug/pretrain_final_encoder.pth \
  --data_location /scratch/hakati/spectrograms \
  --train_folds 0 \
  --val_fold 1 \
  --output_dir pretrain_finetune_results \
  --epochs 101 \
  --batch_size 8 \
  --device cuda \
  --num_workers 4

