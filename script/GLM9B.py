from modelscope import snapshot_download

model_dir = snapshot_download(
    'ZhipuAI/glm-4-9b-chat',
    cache_dir='/hy-tmp/model_cache/',
    revision='master')


# chmod +x start_api_server.sh
# ./start_api_server.sh