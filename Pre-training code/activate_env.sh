#!/usr/bin/env bash
# Activate the eeg2025 micromamba env with the libstdc++ (GLIBCXX) fix baked in.
# Usage:  source ~/EEG_challenge/env/activate_env.sh
if ! command -v micromamba >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load micromamba 2>/dev/null || true
    fi
fi
if ! command -v micromamba >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v micromamba >/dev/null 2>&1; then
    echo "[WARN] micromamba still not found on PATH." >&2
    return 1 2>/dev/null || exit 1
fi

eval "$(micromamba shell hook -s bash)"
micromamba activate eeg2025

module unload gcc/10.2.0 2>/dev/null || true

# The actual GLIBCXX crash fix: force the env's newer libstdc++ ahead of the
# stray system one that shadows it. Without this, importing torchvision throws
# a GLIBCXX_... not found error.
export LD_PRELOAD="$HOME/micromamba/envs/eeg2025/lib/libstdc++.so.6"

echo "[OK] eeg2025 env active: $(which python)"
