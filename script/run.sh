#!/bin/bash

# 启动Qwen服务
start_service_qwen() {
    tmux new-session -d -s mysession "cd /hy-tmp/script/ && conda activate myenv && sh start_qwen_api_server.sh; exec bash"
}

# 启动Qwen执行的脚本
start_script_main() {
    tmux new-window -t mysession:1 "sleep 60s && cd /hy-tmp/script/ && conda activate myenv && python preType1-ChangShiCuoWu-BuWeiCuoWu.py && python preType345-MaoDun.py; exec bash"
    [return int;]
}

# 启动GLM的服务
start_service_glm() {
    tmux new-window -t mysession:2 "cd /hy-tmp/script/ && conda activate myenv && sh start_glm_api_server.sh; exec bash"
}

# 启动GLM执行的脚本
start_script_main_2() {
    tmux new-window -t mysession:3 "sleep 60s && cd /hy-tmp/script/ && conda activate myenv && python Type1-ChangShiCuoWu-BuWeiCuoWu-2Model.py && python Type1-ChangShiCuoWu-ShiJianCuoWu.py && python Type2-ShuZhiDanWeiCuoWu.py && python Type345-MaoDun-2Model.py && python Type7-JiSuanCuoWu.py && python Type8-YuJuChongFu-JiXie+LLM.py; exec bash"
}

# 启动GLM执行的脚本
start_script_main_linshi() {
    tmux new-window -t mysession:3 "
    sleep 60s && cd /hy-tmp/script/ && conda activate myenv && python Type1-ChangShiCuoWu-BuWeiCuoWu-2Model.py && python Type345-MaoDun-2Model.py; exec bash
    "
}

# 终止Qwen服务
stop_service_qwen() {
    tmux send-keys -t mysession C-c
}

# 关闭所有 tmux 窗口和会话
cleanup() {
    tmux kill-session -t mysession
}

## 启动第一个服务
#start_service_qwen
## 启动第二个标签页的脚本
#start_script_main

## 等待第二个脚本完成
#echo "等待第二个脚本完成..."
#tmux wait-for -S main_script_done

## 终止第一个标签页的服务
#stop_service_qwen
#
# 启动第三个标签页的服务
start_service_glm

echo "启动完毕GLM的服务"
#
## 等待第三个服务启动并输出 INFO
#echo "等待第三个服务启动..."
#tmux wait-for -S glm_running
#
# 启动第四个标签页的脚本
#start_script_main_2
#
## 等待第四个脚本完成
#echo "等待第四个脚本完成..."
#tmux wait-for -S main_script_2_done


start_script_main_linshi
# 关闭所有 tmux 窗口和会话
cleanup

echo "所有任务完成并已清理。"
