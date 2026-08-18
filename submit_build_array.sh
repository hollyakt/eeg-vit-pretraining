#!/bin/bash
#SBATCH --job-name=eeg_spec
#SBATCH --account=anwargrp
#SBATCH --partition=cpu
#SBATCH --array=0-199%64
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/spec_%A_%a.out
#SBATCH --error=logs/spec_%A_%a.err
set -euo pipefail
PROJECT=~/EEG_challenge
cd "$PROJECT"
mkdir -p logs /scratch/hakati/spectrograms
set +u; source "$PROJECT/env/activate_env.sh"; set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
NUM_SHARDS=200
echo "Node $(hostname) | shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
python build_pretrain_data.py \
    --raw_dir   "/scratch/hakati/raw" \
    --out_dir   "/scratch/hakati/spectrograms" \
    --pattern   "**/*.bdf" \
    --window_sec 30 --overlap 0.5 \
    --dtype     float16 \
    --num_shards ${NUM_SHARDS} \
    --shard_id   ${SLURM_ARRAY_TASK_ID}
echo "shard ${SLURM_ARRAY_TASK_ID} done."
