#!/bin/bash
# Batch FULL MODEL inference startup script
# Run in background with nohup, output redirected to log file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/batch_inference_full_${TIMESTAMP}.log"

echo "Starting batch FULL MODEL inference task..."
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor progress: tail -f $LOG_FILE"
echo "View process: ps aux | grep run_all_full.py"
echo ""

# Run in background with nohup
# Use yifei environment to run main control script
# Note: tensor-parallel-size is auto-calculated from gpu_ids
#
# 数据集配置选项（三选一）:
# 1. 使用默认AIME数据集 (不传任何数据集参数)
# 2. 指定数据集名称: --dataset-name "math-ai/aime25" --dataset-split "test"
# 3. 使用配置文件: --dataset-config /path/to/dataset_config.json
#
# Base Model 测试选项:
# - 添加 --test-base-model 参数可在测试 checkpoints 前先测试 base model
# - 同时需要指定 --base-model-path 参数指向原始未微调模型路径
#
nohup conda run -n yifei --no-capture-output python run_all_full.py \
  --base-dir ~/yifei/LIMO/train/examples/saves/qwen_3_32b/ \
  --output-base /root/auto_sync_back/aime_results/qwen3_3_32b_sft/ \
  --gpu_ids "2,3,4,5" \
  --port 8001 \
  --concurrency 240 \
  --num-samples 8 \
  --checkpoints-per-group 1 \
  --reasoning-parser qwen3 \
  --vllm-conda-env vllm \
  --infer-conda-env yifei \
  --skip-existing \
  > "$LOG_FILE" 2>&1 &
#  --dataset-config /path/to/your/dataset_config.json \  # 取消注释以使用自定义数据集配置
#  --dataset-name "your-dataset-name" \                    # 或取消注释以指定数据集名称
#  --dataset-split "test" \         
#   --test-base-model \
#   --base-model-path "Qwen/Qwen3-1.7B" \                       # 和数据集分割

PID=$!
echo "✓ Task started (PID: $PID)"
echo ""
echo "Stop task: kill $PID"
echo ""

# Save PID to file
echo "$PID" > "${LOG_DIR}/batch_inference_full.pid"
echo "PID saved to: ${LOG_DIR}/batch_inference_full.pid"

