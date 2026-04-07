#!/bin/bash
# ============================================================================
# run_paligemma_pi0.sh
# VLM → VLA spatial probing: PaliGemma (base) vs π0 (fine-tuned)
#
# Pair:
#   Base VLM : paligemma/3b-pt  (google/paligemma-3b-pt-224)
#   VLA      : pi0/base         (lerobot/pi0_base)
#
# Conda env : qwen3  (transformers >= 4.41 for PaliGemma, has seaborn/sklearn)
# GPUs      : GPU 2 (paligemma), GPU 3 (pi0)  — each model ~6-8 GB
#
# Note on π0: PaliGemma backbone inside π0 is probed via VQA-style inference.
#   The π0 model generates robot actions, not text. Accuracy metrics will be
#   N/A for π0; focus the comparison on the hidden-state delta analysis.
#
# Usage:
#   bash run_paligemma_pi0.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(dirname "$SCRIPT_DIR")/probing.py"
CONDA_ENV="qwen3"
PYTHON="conda run --no-capture-output -n $CONDA_ENV python"

STDOUT_LOG_DIR="$(dirname "$SCRIPT_DIR")/results/logs"
mkdir -p "$STDOUT_LOG_DIR"

# ── Step 0: Ensure lerobot is installed (needed for π0) ──────────────────────
echo "=== Checking/installing lerobot in '$CONDA_ENV' ==="
if conda run --no-capture-output -n $CONDA_ENV python -c "import lerobot" 2>/dev/null; then
    echo "lerobot already installed."
else
    echo "Installing lerobot…"
    conda run --no-capture-output -n $CONDA_ENV pip install -q lerobot
    echo "lerobot installed."
fi
echo ""

# ── Step 1: Run inference in parallel ─────────────────────────────────────────
echo "==================================================="
echo " Group 2: PaliGemma vs π0 — Inference"
echo " PaliGemma 3b-pt → GPU 2"
echo " π0 base         → GPU 3"
echo "==================================================="

PIDS=()

# --- PaliGemma (base VLM) ---
log_paligemma="$STDOUT_LOG_DIR/paligemma_3b-pt_stdout.log"
echo "[GPU 2] paligemma/3b-pt → $log_paligemma"
CUDA_VISIBLE_DEVICES=2 $PYTHON "$SCRIPT" \
    --model_type paligemma \
    --scales 3b-pt \
    > "$log_paligemma" 2>&1 &
PIDS+=($!)

# --- π0 (VLA fine-tuned from PaliGemma) ---
log_pi0="$STDOUT_LOG_DIR/pi0_base_stdout.log"
echo "[GPU 3] pi0/base → $log_pi0"
CUDA_VISIBLE_DEVICES=3 $PYTHON "$SCRIPT" \
    --model_type pi0 \
    --scales base \
    > "$log_pi0" 2>&1 &
PIDS+=($!)

echo ""
echo "Waiting for inference to complete (both models running in parallel)…"
FAILED=0
LABELS=("paligemma/3b-pt" "pi0/base")
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${LABELS[$i]}"
    if wait $pid; then
        echo "[DONE] $label (PID $pid)"
    else
        echo "[FAIL] $label (PID $pid) — exit code $?"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "WARNING: $FAILED model(s) failed. Check logs:"
    echo "  $log_paligemma"
    echo "  $log_pi0"
    echo "Skipping merge step."
    exit 1
fi

# ── Step 2: Merge — generate before/after comparison plots ───────────────────
echo ""
echo "==================================================="
echo " Group 2: Generating VLM→VLA comparison plots"
echo "==================================================="
merge_log="$STDOUT_LOG_DIR/paligemma_pi0_merge_stdout.log"
$PYTHON "$SCRIPT" \
    --model_type paligemma_pi0 \
    --merge \
    --group-name paligemma_pi0 \
    2>&1 | tee "$merge_log"

echo ""
echo "==================================================="
echo " ALL DONE: paligemma_pi0"
echo " Results : $(dirname "$SCRIPT_DIR")/results/saved_data/"
echo " Compare : $(dirname "$SCRIPT_DIR")/results/compare/paligemma_pi0/"
echo "==================================================="
