#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2

CONFIG_FILE="train_limo_try.yaml"
export DISABLE_VERSION_CHECK=1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/limo_train_$(date +%Y%m%d_%H%M%S).log"

nohup bash -c "export PYTHONPATH=../src:\$PYTHONPATH && python -m llamafactory.cli train ${CONFIG_FILE}" > ${LOG_FILE} 2>&1 &

PID=$!
echo "Training started with PID: ${PID}"
echo "Log file: ${LOG_FILE}"

tail -f ${LOG_FILE}
