# EEG ViT Pretraining Pipeline

Self-supervised (MAE) pretraining of a ViT-Small backbone on HBN-EEG spectrograms,
for the NeurIPS 2025 EEG Foundation Challenge (reaction-time and p-factor regression).
The backbone is MAE-pretrained, then frozen while a per-task regression head is fine-tuned.

## Pipeline
1. Download data (HBN-EEG, 100 Hz, BDF) from S3:
   `aws s3 cp --recursive --no-sign-request s3://nmdatasets/NeurIPS25/R1_L100_bdf ./raw/R1_L100_bdf`
2. Generate spectrograms:
   `python build_pretrain_data.py --raw_dir raw/R1_L100_bdf --out_dir spectrograms/ --window_sec 30 --overlap 0.5 --dtype float16`
   (one .npz per recording, key `spectrograms`, shape (128, n_windows, 224, 224))
3. Pretrain: `python pretrain_vit.py --data_dir spectrograms/ --output_dir checkpoints/ --model vit_small --epochs 100 --batch_size 16`
4. Fine-tune per task (Model 1 = reaction time, Model 2 = p-factor) via `eval_finetune_*.py`.

## Environment
Conda/micromamba env from `environment.yml` (torch 2.6.0+cu124, mne, scipy, timm, ...).
Activate: `source env/activate_env.sh` (includes the LD_PRELOAD fix for the cluster GLIBCXX issue).

## Data notes
HBN-EEG, 100 Hz, band-limited 0.5-50 Hz, 128 EEG channels (+ Cz reference).
Releases R1-R11; R6 = validation, R12 = internal test (unreleased).

## Fixes applied vs. original code
- `preprocessing.py`: skip mains notch when line freq >= Nyquist / >= low-pass (crashed at 100 Hz).
- `vision_transformer.py`: forward() reconstructs from the last block by default (was leaving `recons` as int 0 -> LayerNorm crash).
- `pretrain_vit.py`: create output dir before saving the final checkpoint.

## Known TODO
- `datasets/ECOG90S_dataloader` used by fine-tuning scripts is not yet provided.
- `dataloader.py` splits by channel/window; switch to participant-level CV to avoid leakage.
