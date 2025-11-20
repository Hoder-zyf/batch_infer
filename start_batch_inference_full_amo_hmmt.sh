#!/bin/bash
# Batch FULL MODEL inference startup script - AMO Bench & HMMT Nov
# Run in background with nohup, output redirected to log file
# 依次推理: 1. AMO Bench -> 2. HMMT Nov

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "================================================"
echo "Starting Sequential Batch Inference Tasks"
echo "Task 1: AMO Bench (with base model baseline)"
echo "Task 2: HMMT Nov (with base model baseline)"
echo "Samples per question: 4"
echo "Base model: Qwen/Qwen3-32B"
echo "================================================"
echo ""

# ============================================
# Task 1: AMO Bench
# ============================================
echo "[Task 1/2] Starting AMO Bench inference..."
LOG_FILE_AMO="${LOG_DIR}/batch_inference_amo_${TIMESTAMP}.log"
echo "  Log file: $LOG_FILE_AMO"

nohup conda run -n yifei --no-capture-output python run_all_full.py \
  --base-dir ~/yifei/LIMO/train/examples/saves/qwen_3_32b/ \
  --output-base /root/auto_sync_back/amo_results/qwen3_3_32b_sft/ \
  --gpu_ids "0,1,2,3" \
  --port 8001 \
  --concurrency 240 \
  --num-samples 4 \
  --checkpoints-per-group 1 \
  --reasoning-parser qwen3 \
  --vllm-conda-env vllm \
  --infer-conda-env yifei \
  --skip-existing \
  --dataset-config ./dataset_configs/amo_bench.json \
  --test-base-model \
  --base-model-path "Qwen/Qwen3-32B" \
  > "$LOG_FILE_AMO" 2>&1 &


PID_AMO=$!
echo "  ✓ AMO Bench task started (PID: $PID_AMO)"
echo "  Monitor: tail -f $LOG_FILE_AMO"
echo ""

# Save PID
echo "$PID_AMO" > "${LOG_DIR}/batch_inference_amo.pid"

# Wait for AMO Bench task to complete
echo "  Waiting for AMO Bench task to complete..."
wait $PID_AMO
AMO_EXIT_CODE=$?

if [ $AMO_EXIT_CODE -eq 0 ]; then
    echo "  ✓ AMO Bench task completed successfully!"
else
    echo "  ✗ AMO Bench task failed with exit code: $AMO_EXIT_CODE"
    echo "  Check log: $LOG_FILE_AMO"
    exit $AMO_EXIT_CODE
fi

echo ""
echo "================================================"
echo ""

# ============================================
# Task 2: HMMT Nov
# ============================================
echo "[Task 2/2] Starting HMMT Nov inference..."
LOG_FILE_HMMT="${LOG_DIR}/batch_inference_hmmt_${TIMESTAMP}.log"
echo "  Log file: $LOG_FILE_HMMT"

nohup conda run -n yifei --no-capture-output python run_all_full.py \
  --base-dir ~/yifei/LIMO/train/examples/saves/qwen_3_32b/ \
  --output-base /root/auto_sync_back/hmmt_results/qwen3_3_32b_sft/ \
  --gpu_ids "0,1,2,3" \
  --port 8001 \
  --concurrency 240 \
  --num-samples 4 \
  --checkpoints-per-group 1 \
  --reasoning-parser qwen3 \
  --vllm-conda-env vllm \
  --infer-conda-env yifei \
  --skip-existing \
  --dataset-config ./dataset_configs/hmmt_nov.json \
  --test-base-model \
  --base-model-path "Qwen/Qwen3-32B" \
  > "$LOG_FILE_HMMT" 2>&1 &


PID_HMMT=$!
echo "  ✓ HMMT Nov task started (PID: $PID_HMMT)"
echo "  Monitor: tail -f $LOG_FILE_HMMT"
echo ""

# Save PID
echo "$PID_HMMT" > "${LOG_DIR}/batch_inference_hmmt.pid"

# Wait for HMMT Nov task to complete
echo "  Waiting for HMMT Nov task to complete..."
wait $PID_HMMT
HMMT_EXIT_CODE=$?

if [ $HMMT_EXIT_CODE -eq 0 ]; then
    echo "  ✓ HMMT Nov task completed successfully!"
else
    echo "  ✗ HMMT Nov task failed with exit code: $HMMT_EXIT_CODE"
    echo "  Check log: $LOG_FILE_HMMT"
    exit $HMMT_EXIT_CODE
fi

echo ""
echo "================================================"
echo "All tasks completed!"
echo "================================================"
echo ""
echo "Results:"
echo "  AMO Bench:  /root/auto_sync_back/amo_results/qwen3_3_32b_sft/"
echo "  HMMT Nov:   /root/auto_sync_back/hmmt_results/qwen3_3_32b_sft/"
echo ""
echo "Logs:"
echo "  AMO Bench:  $LOG_FILE_AMO"
echo "  HMMT Nov:   $LOG_FILE_HMMT"
echo ""

