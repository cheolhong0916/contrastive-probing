#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(dirname "$SCRIPT_DIR")/probing.py"
PYTHON="/usr/bin/python3"
MODEL="qwen"

# Logs go to results/logs/ (managed by probing.py)
STDOUT_LOG_DIR="$(dirname "$SCRIPT_DIR")/results/logs"
mkdir -p "$STDOUT_LOG_DIR"

# GPU plan: Qwen ~10GB each
# GPU 5: vanilla  GPU 6: 80k + 400k  GPU 7: 800k + 2m
SCALES=("vanilla" "80k" "400k" "800k" "2m")
GPUS=(5 6 6 7 7)

echo "========================================="
echo " Qwen Contrastive Probing: Launching ${#SCALES[@]} scales in parallel"
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

echo "========================================="
echo " Qwen Contrastive Probing: Running merge"
echo "========================================="
$PYTHON "$SCRIPT" --model_type $MODEL \
    --scales vanilla 80k 400k 800k 2m \
    --merge --group-name qwen \
    2>&1 | tee "${STDOUT_LOG_DIR}/qwen_merge_stdout.log"

echo ""
echo "ALL DONE: $MODEL"
echo "Results: $(dirname "$SCRIPT_DIR")/results/saved_data/qwen_*/"
echo "Compare: $(dirname "$SCRIPT_DIR")/results/compare/qwen/"
