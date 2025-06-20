#!/bin/bash

SHELL_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="meta-llama/Meta-Llama-3.1-70B"
# MODEL_CONFIG_NAME="4-Func-mail"
MODEL_CONFIG_NAME="48-Perf-mail"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

export MODEL_NAME
export MODEL_CONFIG_NAME

USE_COMPILE="True"
export USE_COMPILE
PROFILE_DIR="$SHELL_DIR/logs/${MODEL_CONFIG_NAME}_COMPILE_${USE_COMPILE}/${TIMESTAMP}"
mkdir -p "$PROFILE_DIR"
export PROFILE_DIR

echo "Running with USE_COMPILE=$USE_COMPILE"
START_TIME=$(date +%s)
ONEDNN_VERBOSE=all python -u llama31_inner_profile.py >> "$PROFILE_DIR/onednn.verbose.log" 2>&1
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
python process_log.py "$PROFILE_DIR"
echo "Execution time (USE_COMPILE=$USE_COMPILE): ${DURATION}s" | tee -a "$PROFILE_DIR/onednn.verbose.log"

USE_COMPILE="False"
export USE_COMPILE
PROFILE_DIR="$SHELL_DIR/logs/${MODEL_CONFIG_NAME}_COMPILE_${USE_COMPILE}/${TIMESTAMP}"
mkdir -p "$PROFILE_DIR"
export PROFILE_DIR

echo "Running with USE_COMPILE=$USE_COMPILE"
START_TIME=$(date +%s)
ONEDNN_VERBOSE=all python -u llama31_inner_profile.py >> "$PROFILE_DIR/onednn.verbose.log" 2>&1
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
python process_log.py "$PROFILE_DIR"
echo "Execution time (USE_COMPILE=$USE_COMPILE): ${DURATION}s" | tee -a "$PROFILE_DIR/onednn.verbose.log"
