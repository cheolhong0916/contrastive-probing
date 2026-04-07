#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(dirname "$SCRIPT_DIR")/probing.py"
PYTHON="conda run --no-capture-output -n vila python"

# Logs go to results/logs/ (managed by probing.py)
STDOUT_LOG_DIR="$(dirname "$SCRIPT_DIR")/results/logs"
mkdir -p "$STDOUT_LOG_DIR"

# 6 models to run for nvila_synth_compare:
#   vanilla / 80k / 400k             → model_type=nvila          (fine-tuned baselines)
#   80k-5pct / 80k-10pct / 400k-5pct → model_type=nvila_synthetic (synthetic-mix models)
#
# GPU assignment (NVILA ~8GB each):
#   GPU 0: nvila/vanilla         GPU 1: nvila/80k
#   GPU 2: nvila_synthetic/80k-5pct  GPU 3: nvila_synthetic/80k-10pct
#   GPU 4: nvila/400k            GPU 5: nvila_synthetic/400k-5pct

declare -a MODEL_TYPES=("nvila"    "nvila"  "nvila_synthetic" "nvila_synthetic" "nvila"   "nvila_synthetic" "nvila"   "nvila_synthetic")
declare -a SCALES=(     "vanilla"  "80k"    "80k-5pct"        "80k-10pct"       "400k"    "400k-5pct"       "800k"    "800k-5pct")
declare -a GPUS=(        0          1        2                  3                  4         5              6          7)

echo "========================================="
echo " NVILA-Synthetic Mix: Launching models in parallel"
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

echo "========================================="
echo " NVILA-Synthetic Mix: Merge (vanilla / 80k / 80k-5pct / 80k-10pct / 400k / 400k-5pct / 800k / 800k-5pct)"
echo "========================================="
$PYTHON "$SCRIPT" --model_type nvila_synth_compare \
    --merge --group-name nvila_synth_compare \
    2>&1 | tee "${STDOUT_LOG_DIR}/nvila_synth_compare_merge_stdout.log"

echo ""
echo "ALL DONE"
echo "Results: $(dirname "$SCRIPT_DIR")/results/saved_data/{nvila,nvila_synthetic}_*/"
echo "Compare: $(dirname "$SCRIPT_DIR")/results/compare/nvila_synth_compare/"
