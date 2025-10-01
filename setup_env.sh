#!/bin/zsh

set -euo pipefail

PROJECT_ROOT="/Users/luorix/Desktop/MetaMobility Lab (CMU)/projects/F-2025/transfer-learning"
ENV_NAME="transfer-learning"
ENV_FILE="$PROJECT_ROOT/environment.yml"
OPEN_SIM_DIR="$PROJECT_ROOT/processing/opensim"

# Ensure conda is available in this shell and initialize zsh hook
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.zsh hook)"
else
  # Try common install locations
  if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    eval "$(conda shell.zsh hook)"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    eval "$(conda shell.zsh hook)"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    eval "$(conda shell.zsh hook)"
  else
    echo "Conda not found. Please install Miniforge/Miniconda and ensure conda is on PATH." >&2
    exit 1
  fi
fi

# Create or update the environment
if conda env list | grep -E "^$ENV_NAME\s" >/dev/null 2>&1; then
  echo "Environment $ENV_NAME already exists. Updating from $ENV_FILE..."
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "Creating environment $ENV_NAME from $ENV_FILE..."
  conda env create -f "$ENV_FILE"
fi

# Activate the environment in this zsh session
conda activate "$ENV_NAME"

# Install processing/opensim via setup.py (editable)
if [ -f "$OPEN_SIM_DIR/setup.py" ]; then
  echo "Installing processing/opensim via setup.py (editable)..."
  python -m pip install --no-cache-dir --upgrade pip
  python -m pip install -e "$OPEN_SIM_DIR"
else
  echo "setup.py not found in $OPEN_SIM_DIR. Skipping local install." >&2
fi

echo "Environment $ENV_NAME is ready and activated."
