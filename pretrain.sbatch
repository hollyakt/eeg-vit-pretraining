#!/bin/bash
#SBATCH --job-name=eeg_pretrain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=pretrain_%j.log

set -e
source ~/EEG_challenge/env/activate_env.sh
cd ~/EEG_challenge

python -u pretrain_vit.py \
  --data_dir  ~/EEG_challenge/spectrograms_pretrain \
  --output_dir ~/EEG_challenge/pretrain_checkpoints \
  --model vit_small \
  --epochs 100 --batch_size 16 --num_workers 4 \
  --lr 1e-4 --warmup_epochs 5 \
  --save_interval 10 --log_interval 50
