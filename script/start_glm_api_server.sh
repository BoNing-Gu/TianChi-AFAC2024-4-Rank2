#!/bin/bash

# Define variables
# export MODELSCOPE_CACHE=/hy-tmp/model_cache/
# export VLLM_USE_MODELSCOPE=True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
MODEL_PATH="/hy-tmp/model_cache/ZhipuAI/glm-4-9b-chat"
SERVED_MODEL_NAME="glm-4-9b-chat"
PORT=8000
HOST="0.0.0.0"
GPU_MEMORY_UTILIZATION=1
MAX_MODEL_LEN=10240

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