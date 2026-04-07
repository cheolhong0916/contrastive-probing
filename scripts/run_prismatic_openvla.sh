#!/bin/bash
# ============================================================================
# run_prismatic_openvla.sh
# VLM → VLA spatial probing: PrismaticVLM (base) vs OpenVLA (fine-tuned)
#
# Pair:
#   Base VLM : prismatic/7b-dino  (native prismatic package ID)
#   VLA      : openvla/7b         (openvla/openvla-7b, locally cached)
#
# Conda envs:
#   Prismatic → prismatic-vlms  (native prismatic package, timm, flash_attn)
#   OpenVLA   → qwen3           (transformers 4.57, timm installed separately)
#
# GPUs: GPU 0 (prismatic), GPU 1 (openvla)  — each model ~15-20 GB
#
# Usage:
#   bash run_prismatic_openvla.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(dirname "$SCRIPT_DIR")/probing.py"

# Local HF cache for OpenVLA (already downloaded)
export HF_HUB_DIR="/data/shared/Qwen/mydisk/huggingface/hub"

STDOUT_LOG_DIR="$(dirname "$SCRIPT_DIR")/results/logs"
mkdir -p "$STDOUT_LOG_DIR"

# ── Step 0: Ensure dependencies ───────────────────────────────────────────────
echo "=== Checking dependencies ==="
conda run --no-capture-output -n prismatic-vlms pip install -q scikit-learn seaborn 2>&1 | tail -1
echo "Dependencies OK."
echo ""

# ── Step 1: Run inference in parallel ─────────────────────────────────────────
echo "==================================================="
echo " Group 1: Prismatic VLM vs OpenVLA — Inference"
echo " Prismatic 7b-dino → GPU 0  (prismatic-vlms env)"
echo " OpenVLA   7b      → GPU 1  (qwen3 env)"
echo "==================================================="

PIDS=()

# --- Prismatic VLM (base) — uses native prismatic package ---
log_prismatic="$STDOUT_LOG_DIR/prismatic_7b-dino_stdout.log"
echo "[GPU 0] prismatic/7b-dino → $log_prismatic"
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n prismatic-vlms python "$SCRIPT" \
    --model_type prismatic \
    --scales 7b-dino \
    > "$log_prismatic" 2>&1 &
PIDS+=($!)

# --- OpenVLA (VLA fine-tuned from prismatic/7b-dino) — qwen3 + local cache ---
log_openvla="$STDOUT_LOG_DIR/openvla_7b_stdout.log"
echo "[GPU 1] openvla/7b → $log_openvla"
CUDA_VISIBLE_DEVICES=1 HF_HUB_DIR="$HF_HUB_DIR" \
    conda run --no-capture-output -n qwen3 python "$SCRIPT" \
    --model_type openvla \
    --scales 7b \
    > "$log_openvla" 2>&1 &
PIDS+=($!)

echo ""
echo "Waiting for inference to complete (both models running in parallel)…"
FAILED=0
LABELS=("prismatic/7b-dino" "openvla/7b")
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
    echo "  $log_prismatic"
    echo "  $log_openvla"
    echo "Skipping merge step."
    exit 1
fi

# ── Step 2: Merge — generate before/after comparison plots ───────────────────
echo ""
echo "==================================================="
echo " Group 1: Generating VLM→VLA comparison plots"
echo "==================================================="
merge_log="$STDOUT_LOG_DIR/prismatic_openvla_merge_stdout.log"
conda run --no-capture-output -n qwen3 python "$SCRIPT" \
    --model_type prismatic_openvla \
    --merge \
    --group-name prismatic_openvla \
    2>&1 | tee "$merge_log"

echo ""
echo "==================================================="
echo " ALL DONE: prismatic_openvla"
echo " Results : $(dirname "$SCRIPT_DIR")/results/saved_data/"
echo " Compare : $(dirname "$SCRIPT_DIR")/results/compare/prismatic_openvla/"
echo "==================================================="
