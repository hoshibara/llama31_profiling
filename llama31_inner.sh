#!/bin/bash

SHELL_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="meta-llama/Meta-Llama-3.1-70B"
MODEL_CONFIG_NAME="4-Func-mail"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

export MODEL_NAME
export MODEL_CONFIG_NAME

USE_COMPILE="True"
export USE_COMPILE
PROFILE_DIR="$SHELL_DIR/logs/${MODEL_CONFIG_NAME}_COMPILE_${USE_COMPILE}/${TIMESTAMP}"
mkdir -p "$PROFILE_DIR"
export PROFILE_DIR

# 然后再调用你的 Python 脚本
ONEDNN_VERBOSE=all python -u llama31_inner_profile.py >> "$PROFILE_DIR/onednn.verbose.log" 2>&1


USE_COMPILE="False"
export USE_COMPILE
PROFILE_DIR="$SHELL_DIR/logs/${MODEL_CONFIG_NAME}_COMPILE_${USE_COMPILE}/${TIMESTAMP}"
mkdir -p "$PROFILE_DIR"
export PROFILE_DIR

ONEDNN_VERBOSE=all python -u llama31_inner_profile.py >> "$PROFILE_DIR/onednn.verbose.log" 2>&1
