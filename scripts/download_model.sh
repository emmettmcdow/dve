#!/usr/bin/env bash
# Download a dve-compatible embedding model.
# Usage: ./scripts/download_model.sh <model>
#
# Available models:
#   mpnet  - sentence-transformers/all-mpnet-base-v2 (recommended, higher quality)
#
# The model is downloaded to models/<model_name>/ in the repo root.
# C/C++ users: pass this directory to dve_init.
# Swift users: add the downloaded files to your Xcode target's Copy Bundle Resources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

usage() {
    echo "Usage: $0 <model>"
    echo ""
    echo "Available models:"
    echo "  mpnet  - sentence-transformers/all-mpnet-base-v2"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

MODEL="$1"

case "$MODEL" in
    mpnet)
        OUT_DIR="$MODELS_DIR/all_mpnet_base_v2"
        if [ -d "$OUT_DIR/all_mpnet_base_v2.mlpackage" ]; then
            echo "Model already exists at $OUT_DIR. Nothing to do."
            exit 0
        fi
        echo "Downloading all-mpnet-base-v2..."
        echo "This requires Python and the dve model generation script."
        mkdir -p "$MODELS_DIR"
        cd "$MODELS_DIR"
        if [ ! -d "venv" ]; then
            echo "Setting up Python venv..."
            python3 -m venv venv
            ./venv/bin/pip install --quiet torch sentence-transformers coremltools
        fi
        ./venv/bin/python gen-coreml.py \
            sentence-transformers/all-mpnet-base-v2 \
            --output all_mpnet_base_v2
        echo "Done. Model saved to $OUT_DIR"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo ""
        usage
        ;;
esac
