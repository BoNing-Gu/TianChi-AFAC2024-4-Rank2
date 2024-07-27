#!/bin/bash

# Define variables
MODEL_PATH="/hy-tmp/model_cache/ZhipuAI/glm-4-9b-chat"
SERVED_MODEL_NAME="glm-4-9b-chat"
PORT=8000
HOST="0.0.0.0"
GPU_MEMORY_UTILIZATION=1
MAX_MODEL_LEN=12288

# Run the API server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --dtype half \
    --trust-remote-code \
    --port "$PORT" \
    --host "$HOST" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN"
