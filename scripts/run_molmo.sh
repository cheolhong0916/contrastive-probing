#!/bin/bash
# Molmo contrastive probing — vanilla only.
#
# Note: fine-tuned variants (80k/400k/800k/2m) were previously listed in the
# registry, but the corresponding HF repos contain only wandb logs (no model
# weights) and the trained weights are no longer recoverable. If you have a
# local Molmo fine-tune, add it to MODEL_REGISTRY["molmo"]["checkpoints"] in
# probing.py and extend SCALES below.
#
# Env: Molmo's bundled modeling_molmo.py uses the legacy past_key_values
# tuple API; transformers >= 4.42 breaks it. Use a conda env with
# transformers ~= 4.46 (the `vila` env on this machine satisfies that).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(dirname "$SCRIPT_DIR")/probing.py"
# Set PYTHON to the interpreter of the env that satisfies envs/requirements-vila.txt
# (transformers <= 4.46 — Molmo's bundled modeling code uses the legacy
# past_key_values tuple API and breaks under transformers >= 4.42). Examples:
#   PYTHON="python3"
#   PYTHON="conda run --no-capture-output -n probing-vila python"
#   PYTHON="/path/to/.venv-vila/bin/python"
PYTHON="${PYTHON:-python3}"
MODEL="molmo"

STDOUT_LOG_DIR="$(dirname "$SCRIPT_DIR")/results/logs"
mkdir -p "$STDOUT_LOG_DIR"

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
echo "Waiting for ${#PIDS[@]} process(es)..."
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
