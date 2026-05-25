#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(dirname "$SCRIPT_DIR")/probing.py"
PYTHON="conda run --no-capture-output -n vila python"
MODEL="nvila"

# Logs go to results/logs/ (managed by probing.py)
STDOUT_LOG_DIR="$(dirname "$SCRIPT_DIR")/results/logs"
mkdir -p "$STDOUT_LOG_DIR"

# GPU plan: NVILA ~8GB each
SCALES=("vanilla" "80k" "400k" "800k" "2m" "roborefer")
GPUS=(2 3 4 5 6 7)

echo "========================================="
echo " NVILA Contrastive Probing: Launching ${#SCALES[@]} scales in parallel"
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
echo " NVILA Contrastive Probing: Merge 1/2 (without roborefer)"
echo "========================================="
$PYTHON "$SCRIPT" --model_type $MODEL \
    --scales vanilla 80k 400k 800k 2m \
    --merge --group-name nvila \
    2>&1 | tee "${STDOUT_LOG_DIR}/nvila_merge_stdout.log"

echo "========================================="
echo " NVILA Contrastive Probing: Merge 2/2 (with roborefer)"
echo "========================================="
$PYTHON "$SCRIPT" --model_type $MODEL \
    --scales vanilla 80k 400k 800k 2m roborefer \
    --merge --group-name nvila_with_roborefer \
    2>&1 | tee "${STDOUT_LOG_DIR}/nvila_with_roborefer_merge_stdout.log"

echo ""
echo "ALL DONE: $MODEL"
echo "Results: $(dirname "$SCRIPT_DIR")/results/saved_data/nvila_*/"
echo "Compare: $(dirname "$SCRIPT_DIR")/results/compare/nvila/"
echo "Compare: $(dirname "$SCRIPT_DIR")/results/compare/nvila_with_roborefer/"
