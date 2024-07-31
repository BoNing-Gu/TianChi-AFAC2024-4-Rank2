#!/bin/bash

cd /hy-tmp/script/  &&
conda activate myenv &&
python preType1-ChangShiCuoWu-BuWeiCuoWu.py -v final-test  &&
conda deactivate

echo "金融知识补充任务完成。"