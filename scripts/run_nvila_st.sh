#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(dirname "$SCRIPT_DIR")/probing.py"
PYTHON="conda run --no-capture-output -n vila python"

# Logs go to results/logs/ (managed by probing.py)
STDOUT_LOG_DIR="$(dirname "$SCRIPT_DIR")/results/logs"
mkdir -p "$STDOUT_LOG_DIR"

# 3 checkpoints from MCQ-synthetic-trained NVILA run:
#   80k-st  → checkpoint-1250   (80k training steps)
#   400k-st → checkpoint-6250   (400k training steps)
#   800k-st → checkpoint-12500  (800k training steps)
#
# All from: NVILA-Lite-2B-SYNTHETIC_MIX_MCQ_5PCT_2M-20260302_030354
#
# GPU assignment (NVILA ~8GB each):
#   GPU 0: nvila_st/80k-st
#   GPU 1: nvila_st/400k-st
#   GPU 2: nvila_st/800k-st

declare -a MODEL_TYPES=("nvila_st" "nvila_st" "nvila_st")
declare -a SCALES=(     "80k-st"   "400k-st"  "800k-st")
declare -a GPUS=(        0          1          2)

echo "========================================="
echo " NVILA-ST: Launching 3 checkpoints in parallel"
echo "========================================="

PIDS=()
for i in "${!SCALES[@]}"; do
    mtype="${MODEL_TYPES[$i]}"
    scale="${SCALES[$i]}"
    gpu="${GPUS[$i]}"
    log="${STDOUT_LOG_DIR}/${mtype}_${scale}_stdout.log"

    echo "[GPU $gpu] ${mtype}/${scale} -> $log"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" \
        --model_type $mtype \
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
    label="${MODEL_TYPES[$i]}/${SCALES[$i]}"
    if wait $pid; then
        echo "[DONE] $label (PID $pid) - SUCCESS"
    else
        echo "[FAIL] $label (PID $pid) - EXIT CODE $?"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED process(es) failed. Check logs in $STDOUT_LOG_DIR"
fi

echo ""
echo "ALL DONE (inference only)"
echo "Results: $(dirname "$SCRIPT_DIR")/results/saved_data/nvila_st_*/"
echo ""
echo "To merge with nvila baseline (vanilla/80k/400k/800k/2m), run:"
echo "  conda run --no-capture-output -n vila python $SCRIPT \\"
echo "      --model_type nvila_st_compare --merge --group-name nvila_st_compare"
