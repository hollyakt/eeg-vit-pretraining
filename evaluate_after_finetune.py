#!/usr/bin/env python
"""
Evaluate the finetuned model checkpoint on the validation fold.
This script expects the finetune job to save checkpoint at
`pretrain_finetune_results/checkpoint_latest.pth` and will write results to
`pretrain_finetune_results/eval_results.json`.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import vision_transformer as vits
from datasets.ECOG90S_dataloader import EEG_Windows_Model2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='pretrain_finetune_results/checkpoint_latest.pth')
    parser.add_argument('--data_location', default='/scratch/hakati/spectrograms')
    parser.add_argument('--val_fold', default='1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    # build model compatible with eval_finetune_model2's expected architecture
    model = vits.vit_small(patch_size=16, img_size=[224], in_chans=128, num_classes=1)
    model = model.to(device)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = None
    if 'model' in ckpt:
        state = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    # strip prefixes if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def build_val_loader(data_location, val_fold, batch_size, num_workers):
    val_dir = os.path.join(data_location, f"f{val_fold}")
    ds = EEG_Windows_Model2(data_location=val_dir, pfactor_csv_path=None, in_chans=128, transform=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return loader


def evaluate(model, loader, device):
    mse = torch.nn.MSELoss(reduction='mean')
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            # loader should return (samples, targets) or only samples depending on dataset
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                samples, t = batch[0].to(device), batch[1].to(device)
            else:
                samples = batch.to(device)
                # no targets available
                t = None
            out = model(samples, classify=True)
            out = out.view(-1)
            if t is not None:
                t = t.view(-1).to(device)
                preds.append(out.cpu().numpy())
                targets.append(t.cpu().numpy())
    result = {}
    if targets:
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        mse_val = float(((preds - targets) ** 2).mean())
        result['mse'] = mse_val
        # additional metrics
        try:
            from sklearn.metrics import r2_score
            result['r2'] = float(r2_score(targets, preds))
        except Exception:
            pass
    return result


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Loading model from', args.checkpoint)
    model = load_model(args.checkpoint, device)
    print('Building validation loader')
    val_loader = build_val_loader(args.data_location, args.val_fold, args.batch_size, args.num_workers)
    print('Running evaluation...')
    start = time.time()
    results = evaluate(model, val_loader, device)
    results['runtime_sec'] = time.time() - start
    out_path = Path(args.checkpoint).parent / 'eval_results.json'
    out_path.write_text(json.dumps(results, indent=2))
    print('Saved results to', out_path)

if __name__ == '__main__':
    main()
