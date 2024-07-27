from modelscope import snapshot_download

model_dir = snapshot_download(
    'TongyiFinance/Tongyi-Finance-14B-Chat-Int4',
    # 'TongyiFinance/Tongyi-Finance-14B-Chat',
    cache_dir='/hy-tmp/model_cache/',
    revision='master')

# export VLLM_USE_MODELSCOPE=True
# chmod +x start_glm_api_server.sh
# ./start_glm_api_server.sh