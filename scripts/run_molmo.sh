#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(dirname "$SCRIPT_DIR")/probing.py"
# Set PYTHON to the interpreter of the env that satisfies Molmo deps
# (transformers, einops, etc.; trust_remote_code is required by Molmo).
# Examples:
#   PYTHON="python3"
#   PYTHON="conda run --no-capture-output -n molmo python"
#   PYTHON="/path/to/.venv-molmo/bin/python"
PYTHON="${PYTHON:-python3}"
MODEL="molmo"

# Logs go to results/logs/ (managed by probing.py)
STDOUT_LOG_DIR="$(dirname "$SCRIPT_DIR")/results/logs"
mkdir -p "$STDOUT_LOG_DIR"

# Only the public base checkpoint (allenai/Molmo-7B-O-0924) is registered.
SCALES=("vanilla")
GPUS=(0)

echo "========================================="
echo " Molmo Contrastive Probing: Launching ${#SCALES[@]} scale(s)"
echo "========================================="

PIDS=()
for i in "${!SCALES[@]}"; do
    scale="${SCALES[$i]}"
    gpu="${GPUS[$i]}"
    log="${STDOUT_LOG_DIR}/${MODEL}_${scale}_stdout.log"

    echo "[GPU $gpu] $MODEL/$scale -> $log"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" \
        --model_type $MODEL \
        --scales $scale \
        --device cuda \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for all ${#PIDS[@]} processes..."
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    scale="${SCALES[$i]}"
    if wait $pid; then
        echo "[DONE] $MODEL/$scale (PID $pid) - SUCCESS"
    else
        echo "[FAIL] $MODEL/$scale (PID $pid) - EXIT CODE $?"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED scale(s) failed. Check logs in $STDOUT_LOG_DIR"
fi

echo ""
echo "ALL DONE: $MODEL"
echo "Results: $(dirname "$SCRIPT_DIR")/results/saved_data/molmo_*/"
