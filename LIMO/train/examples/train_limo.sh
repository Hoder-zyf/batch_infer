#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

CONFIG_FILE="train_limo.yaml"

LOG_DIR="logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/limo_train_$(date +%Y%m%d_%H%M%S).log"

nohup llamafactory-cli train ${CONFIG_FILE} > ${LOG_FILE} 2>&1 &

PID=$!
echo "Training started with PID: ${PID}"
echo "Log file: ${LOG_FILE}"

tail -f ${LOG_FILE}
