#!/usr/bin/env bash
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

# unload AFTER activating / loading the micromamba module, since loading
# micromamba can silently pull gcc/10.2.0 back in as a dependency, which
# shadows the env's own (newer) libstdc++ and breaks torchvision/Pillow
module unload gcc/10.2.0 2>/dev/null || true

echo "[OK] eeg2025 env active: $(which python)"
