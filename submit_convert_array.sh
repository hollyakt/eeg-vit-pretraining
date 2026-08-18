#!/bin/bash
#SBATCH --job-name=npz2npy
#SBATCH --account=anwargrp
#SBATCH --partition=cpu
#SBATCH --array=0-99%50
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/conv_%A_%a.out
#SBATCH --error=logs/conv_%A_%a.err
set -eo pipefail
cd ~/EEG_challenge
mkdir -p logs
set +u; source ~/EEG_challenge/env/activate_env.sh; set -u
NUM_SHARDS=100
python convert_npz_to_npy.py ${NUM_SHARDS} ${SLURM_ARRAY_TASK_ID}
