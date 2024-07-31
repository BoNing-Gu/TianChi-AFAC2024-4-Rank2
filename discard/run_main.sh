#!/bin/bash

# 启动GLM的服务
function start_service_glm() {
    tmux new-session -d -s mysession "
    cd /hy-tmp/script/ &&
    conda activate myenv &&
    sh start_glm_api_server.sh;
    exec bash"
}

# 启动GLM执行的脚本
function start_script_main() {
    tmux new-window -t mysession:1 "
    sleep 120s && cd /hy-tmp/script/ &&
    conda activate myenv &&
    python Type1-ChangShiCuoWu-BuWeiCuoWu-2Model.py -v final-test &&
    python Type1-ChangShiCuoWu-ShiJianCuoWu-OnlyTime.py -v final-test &&
    python Type2-ShuZhiDanWeiCuoWu.py -v final-test &&
    python Type7-JiSuanCuoWu.py -v final-test &&
    python Type8-YuJuChongFu-JiXie+LLM.py -v final-test;
    exec bash"
}

# 终止GLM服务
function stop_service_glm() {
    tmux send-keys -t mysession:0 C-c
}

# 关闭所有 tmux 窗口和会话
function cleanup() {
    tmux kill-session -t mysession
}

# 启动第一个服务
start_service_glm
# 启动第二个标签页的脚本
start_script_main &
# 等待第二个标签页脚本执行完毕
wait $!

# 终止第一个标签页的服务
stop_service_glm

# 关闭所有 tmux 窗口和会话
cleanup

echo "所有任务完成并已清理。"