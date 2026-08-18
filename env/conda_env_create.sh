#!/bin/bash
#SBATCH --job-name=eeg_fine
#SBATCH --output=eeg_fine_%j.log
#SBATCH --partition=ml-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#!/usr/bin/env bash
set -euo pipefail

# ---------- Parse args ----------
YML="./environment.yml"
ENV_NAME="eeg2025"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yml) YML="$2"; shift 2;;
    -n|--name) ENV_NAME="$2"; shift 2;;
    --root) MAMBA_ROOT="$2"; shift 2;;
    -h|--help)
      sed -n '1,60p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ ! -f "$YML" ]]; then
  echo "ERROR: environment file not found: $YML"
  exit 1
fi

# ---------- Ensure micromamba ----------
need_init_hook=false

if ! command -v micromamba >/dev/null 2>&1; then
  echo "[INFO] micromamba not found in PATH. Trying 'module load micromamba'..."
  if command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    module load micromamba 2>/dev/null || true
  fi
fi

if ! command -v micromamba >/dev/null 2>&1; then
  echo "[INFO] micromamba still not found; attempting direct download to $HOME/.local/bin ..."
  mkdir -p "$HOME/.local/bin"
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$HOME/.local/bin/micromamba"
  chmod +x "$HOME/.local/bin/micromamba"
  export PATH="$HOME/.local/bin:$PATH"
  need_init_hook=true
fi

if ! command -v micromamba >/dev/null 2>&1; then
  echo "ERROR: micromamba unavailable and download failed (no internet?)."
  echo "       Ask your HPC admin for the micromamba module, or place micromamba in PATH."
  exit 1
fi

# ---------- Configure root & shell hook ----------
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
mkdir -p "$MAMBA_ROOT_PREFIX"

# Initialize shell hook for this script run
eval "$(micromamba shell hook -s bash)"

# ---------- Config: strict channel priority ----------
micromamba config set channel_priority strict >/dev/null

# ---------- Create or update env ----------
if micromamba info -e | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] Environment '$ENV_NAME' exists. Updating from $YML ..."
  micromamba env update -n "$ENV_NAME" -f "$YML" -y
else
  echo "[INFO] Creating environment '$ENV_NAME' from $YML ..."
  micromamba create -n "$ENV_NAME" -f "$YML" -y
fi

# ---------- Optional: add GPU/NCCL tuning (customize or remove) ----------
ACTIVATE_DIR="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/etc/conda/activate.d"
DEACTIVATE_DIR="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/99_hpc_cuda_nccl.sh" <<'EOF'
# HPC CUDA/NCCL tuning for PyTorch (adjust to your cluster)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# If your cluster uses Infiniband and you know your interface, set it here:
# export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
# If you have issues with loopback or docker bridges, you can exclude them:
# export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}
# For multi-node runs, you might also need:
# export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
EOF

cat > "$DEACTIVATE_DIR/99_hpc_cuda_nccl.sh" <<'EOF'
unset PYTORCH_CUDA_ALLOC_CONF
unset CUDA_LAUNCH_BLOCKING
unset NCCL_DEBUG
# unset NCCL_SOCKET_IFNAME
# unset NCCL_IB_GID_INDEX
EOF

# ---------- Sanity check ----------
echo "[INFO] Activating env and checking torch/torchvision ..."
micromamba run -n "$ENV_NAME" python - <<'PY'
import torch, torchvision, sys
print("torch:", torch.__version__,
      "cuda:", torch.version.cuda,
      "is_available:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
PY

echo "[SUCCESS] Environment '$ENV_NAME' is ready."
echo "To use it now in this shell:"
echo "  eval \"\$(micromamba shell hook -s bash)\""
echo "  micromamba activate $ENV_NAME"

# Optionally persist hook for future logins:
if $need_init_hook; then
  SHELL_RC="${HOME}/.bashrc"
  if ! grep -q 'micromamba shell hook' "$SHELL_RC" 2>/dev/null; then
    echo '[INFO] Adding micromamba shell hook to ~/.bashrc'
    {
      echo ''
      echo '# micromamba shell hook'
      echo 'if command -v micromamba >/dev/null 2>&1; then'
      echo '  eval "$(micromamba shell hook -s bash)"'
      echo 'fi'
    } >> "$SHELL_RC"
  fi
fi
