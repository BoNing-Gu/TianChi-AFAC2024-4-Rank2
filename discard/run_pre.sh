#!/bin/bash

SESSION="mysession"

# 启动Qwen服务
function start_service_qwen() {
    tmux new-session -d -s $SESSION "
    cd /hy-tmp/script/ &&
    conda activate myenv &&
    sh start_qwen_api_server.sh;
    exec bash"
}

# 启动Qwen执行的脚本
function start_script_pre() {
    tmux new-window -t $SESSION "
    cd /hy-tmp/script/ &&
    conda activate myenv &&
    python preType1-ChangShiCuoWu-BuWeiCuoWu.py -v final-test;
    tmux wait -S pre_script_done;
    exec bash"
}

# 终止Qwen服务
function stop_service_qwen() {
    tmux send-keys -t $SESSION:0 C-c
}

# 关闭所有 tmux 窗口和会话
function cleanup() {
    tmux kill-session -t $SESSION
}

# 启动第一个服务
start_service_qwen

# 启动第二个标签页的脚本
start_script_pre &

# 等待第二个标签页脚本执行完毕
tmux wait pre_script_done

# 终止第一个标签页的服务
stop_service_qwen

# 关闭所有 tmux 窗口和会话
cleanup

echo "所有任务完成并已清理。"

