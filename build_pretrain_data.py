#!/usr/bin/env python3
"""
build_pretrain_data.py
======================

Driver script that turns raw HBN-EEG `.bdf` recordings into the `.npz`
spectrogram files consumed by `pretrain_vit.py --data_dir`.

It is the "glue" between three pieces you already have:

    read_raw_bdf (MNE)  ->  preprocessing.preprocess_eeg_for_spectrograms
                        ->  spects.create_individual_channel_spectrograms_dynamic
                        ->  one .npz per recording, key 'spectrograms'

Output contract
---------------
Each output file contains a single array under the key **'spectrograms'** with
shape **(128, n_windows, 224, 224)** -- i.e. CHANNELS FIRST.

    Why channels-first and not the (n_windows, 128, 224, 224) shown in the
    methodology doc?  Because the thing that actually reads these files,
    `EEGSpectrogramDataset` in pretrain_vit.py, branches on
    `specs.shape[0] == 128`, and dataloader.py reads
    `n_channels, n_windows = shape[0], shape[1]`.  The code wins over the doc.

A few provenance keys (`sfreq`, `window_sec`, `ch_names`, `source_file`) are
also stored; the loaders ignore them.

Usage
-----
    python build_pretrain_data.py \
        --raw_dir  ~/EEG_challenge/raw/R1_mini_L100_bdf \
        --out_dir  ~/EEG_challenge/spectrograms_pretrain \
        --window_sec 30 --overlap 0.5

    # then point the (fixed) pretrainer at the result:
    python pretrain_vit.py --data_dir ~/EEG_challenge/spectrograms_pretrain ...

Notes
-----
* This produces *self-supervised pretraining* data.  Pretraining does not use
  labels, so windows are cut continuously across the whole recording (Model-2
  style) rather than stimulus-locked (Model-1 style).  No participants.tsv /
  p-factor / reaction-time matching happens here -- that is a fine-tuning step.
* HBN is US data recorded on a 128-ch EGI net (+ Cz reference = 129 raw
  channels).  Defaults reflect that: line filter 60 Hz, target rate 100 Hz
  (the "L100" mini release), reference channel 'Cz' dropped -> 128 channels.
* `preprocessing.py` expects micro-volts (its 150 uV artifact threshold and the
  dB floor assume that scale).  MNE returns Volts, so we multiply by 1e6.
"""

import argparse
import gc
import sys
import traceback
from pathlib import Path

import numpy as np

# Make sure the sibling modules (preprocessing.py, spects.py) are importable
# regardless of where this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import mne
except ImportError:
    sys.exit(
        "ERROR: `mne` is not installed in this environment.\n"
        "Activate the eeg2025 env (see ~/EEG_challenge/env/activate_env.sh) "
        "or run:  pip install mne"
    )

from preprocessing import preprocess_eeg_for_spectrograms
from spects import create_individual_channel_spectrograms_dynamic


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def load_bdf(bdf_path, target_sfreq, verbose=False):
    """Read a .bdf file, keep EEG channels, resample, return (data_uV, ch_names, sfreq)."""
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")

    # Keep only EEG channels (drop Status/trigger/misc channels that BDF carries).
    try:
        raw.pick("eeg")
    except (ValueError, TypeError):
        # Older MNE API or no channel types set -> fall back to picking by type.
        picks = mne.pick_types(raw.info, eeg=True, stim=False, misc=False)
        if len(picks):
            raw.pick(picks)

    # Resample to the target rate if needed (BDF may be stored at a higher rate).
    orig_sfreq = float(raw.info["sfreq"])
    if target_sfreq and abs(orig_sfreq - target_sfreq) > 1e-6:
        if verbose:
            print(f"    resampling {orig_sfreq:g} Hz -> {target_sfreq:g} Hz")
        raw.resample(target_sfreq, verbose="ERROR")

    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.info["ch_names"])
    data_v = raw.get_data()          # (n_channels, n_samples), Volts
    data_uv = data_v * 1e6           # -> micro-volts, as preprocessing.py expects

    del raw
    gc.collect()
    return data_uv, ch_names, sfreq


def enforce_channel_count(specs, n_channels, source):
    """Force the channel axis (axis 0) to exactly n_channels via pad/truncate."""
    cur = specs.shape[0]
    if cur == n_channels:
        return specs
    if cur > n_channels:
        print(f"    WARNING [{source}]: {cur} channels -> truncating to {n_channels}")
        return specs[:n_channels]
    # cur < n_channels: zero-pad
    print(f"    WARNING [{source}]: {cur} channels -> zero-padding to {n_channels}")
    pad = np.zeros((n_channels - cur, *specs.shape[1:]), dtype=specs.dtype)
    return np.concatenate([specs, pad], axis=0)


def process_one(bdf_path, args):
    """Full pipeline for a single .bdf. Returns output path or None on skip/fail."""
    stem = bdf_path.stem
    out_path = Path(args.out_dir) / f"{stem}.npz"

    if out_path.exists() and not args.overwrite:
        print(f"  SKIP (exists): {out_path.name}")
        return out_path

    print(f"  Processing: {bdf_path.name}")

    # 1) Read + resample + to micro-volts
    data_uv, ch_names, sfreq = load_bdf(bdf_path, args.target_sfreq, args.verbose)
    if args.verbose:
        print(f"    raw picked shape: {data_uv.shape} @ {sfreq:g} Hz")

    # 2) Preprocess (detrend, band-pass, notch, artifact flag, drop reference)
    pre = preprocess_eeg_for_spectrograms(
        data_uv,
        ch_names=ch_names,
        sfreq=sfreq,
        reference_ch=args.reference_ch,
        line_freq=args.line_freq,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        verbose=args.verbose,
    )
    clean = pre["data"]  # (n_channels, n_samples)
    if args.verbose:
        print(f"    after preprocessing: {clean.shape}")

    # 3) Spectrograms via sliding windows across the whole recording
    window_size = int(round(args.window_sec * sfreq))  # samples
    if clean.shape[1] < window_size:
        print(f"    SKIP: recording {clean.shape[1]} samples < window {window_size}")
        return None

    result = create_individual_channel_spectrograms_dynamic(
        clean,
        fs=sfreq,
        window_size=window_size,
        target_size=(224, 224),
        overlap_percent=args.overlap,
        frequency_focus=True,
        max_freq=args.high_freq,
        min_freq=args.low_freq,
        verbose=args.verbose,
    )

    # `all_spectrograms` is a list [n_channels] of lists [n_windows] of (224,224).
    # Every channel is windowed identically, so this stacks cleanly.
    all_specs = result["all_spectrograms"]
    n_win = min((len(ch) for ch in all_specs), default=0)
    if n_win == 0:
        print(f"    SKIP: no full windows produced")
        return None

    # Trim to the common window count (defensive) and optionally cap.
    if args.max_windows and n_win > args.max_windows:
        n_win = args.max_windows
    specs = np.stack(
        [np.stack(ch[:n_win], axis=0) for ch in all_specs], axis=0
    )  # (n_channels, n_win, 224, 224)

    # 4) Force exactly 128 channels, cast, save
    specs = enforce_channel_count(specs, args.n_channels, stem)
    specs = specs.astype(np.dtype(args.dtype))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        spectrograms=specs,                       # (128, n_win, 224, 224)
        sfreq=np.float32(sfreq),
        window_sec=np.float32(args.window_sec),
        ch_names=np.array(pre["ch_names"], dtype=object),
        source_file=str(bdf_path),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"    -> {out_path.name}  shape={specs.shape}  dtype={specs.dtype}  ({size_mb:.1f} MB)")

    del data_uv, clean, result, all_specs, specs
    gc.collect()
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Build ViT pretraining spectrogram .npz files from raw .bdf EEG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_dir", required=True, help="Root dir to search for .bdf files (recursive).")
    p.add_argument("--out_dir", required=True, help="Where to write .npz spectrogram files.")
    p.add_argument("--pattern", default="**/*.bdf", help="Glob (relative to raw_dir) for input files.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N files (debug).")

    # Signal / preprocessing
    p.add_argument("--target_sfreq", type=float, default=100.0,
                   help="Resample to this rate (Hz). HBN 'L100' mini release is 100 Hz.")
    p.add_argument("--line_freq", type=float, default=60.0, help="Mains notch (Hz). US HBN = 60.")
    p.add_argument("--low_freq", type=float, default=0.5, help="High-pass cutoff (Hz).")
    p.add_argument("--high_freq", type=float, default=50.0, help="Low-pass cutoff / max spectro freq (Hz).")
    p.add_argument("--reference_ch", default="Cz", help="Reference channel name to drop if present.")

    # Windowing
    p.add_argument("--window_sec", type=float, default=30.0,
                   help="Sliding-window length in seconds (30/45 = Model-2 style).")
    p.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction [0,1).")
    p.add_argument("--max_windows", type=int, default=None,
                   help="Cap windows per file (limits .npz size / memory).")

    # Output
    p.add_argument("--n_channels", type=int, default=128, help="Enforced channel-axis size.")
    p.add_argument("--dtype", default="float32", choices=["float16", "float32"],
                   help="Storage dtype. float16 halves disk; loader re-normalizes anyway.")
    p.add_argument("--overwrite", action="store_true", help="Reprocess even if output exists.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir).expanduser()
    args.out_dir = str(Path(args.out_dir).expanduser())
    if not raw_dir.exists():
        sys.exit(f"ERROR: raw_dir does not exist: {raw_dir}")

    bdf_files = sorted(raw_dir.glob(args.pattern))
    if args.limit:
        bdf_files = bdf_files[: args.limit]

    print("=" * 78)
    print("build_pretrain_data.py")
    print(f"  raw_dir     : {raw_dir}")
    print(f"  out_dir     : {args.out_dir}")
    print(f"  found       : {len(bdf_files)} .bdf files (pattern {args.pattern!r})")
    print(f"  window      : {args.window_sec}s @ {args.overlap:.0%} overlap")
    print(f"  resample    : {args.target_sfreq:g} Hz   notch {args.line_freq:g} Hz   "
          f"band {args.low_freq}-{args.high_freq} Hz")
    print(f"  output      : key 'spectrograms', shape ({args.n_channels}, n_windows, 224, 224), "
          f"dtype {args.dtype}")
    print("=" * 78)

    if not bdf_files:
        sys.exit("No .bdf files found. Check --raw_dir / --pattern.")

    ok, skipped, failed = [], [], []
    for i, bdf in enumerate(bdf_files, 1):
        print(f"[{i}/{len(bdf_files)}]")
        try:
            res = process_one(bdf, args)
            (ok if res is not None else skipped).append(bdf.name)
        except Exception as e:  # noqa: BLE001 - keep going on a bad file
            failed.append(bdf.name)
            print(f"  ERROR on {bdf.name}: {e}")
            if args.verbose:
                traceback.print_exc()

    print("\n" + "=" * 78)
    print(f"Done. wrote {len(ok)}  |  skipped {len(skipped)}  |  failed {len(failed)}")
    if failed:
        print("Failed files:", ", ".join(failed))
    print(f"Spectrograms in: {args.out_dir}")
    print("Next: python pretrain_vit.py --data_dir", args.out_dir, "--model vit_small ...")
    print("=" * 78)


if __name__ == "__main__":
    main()
