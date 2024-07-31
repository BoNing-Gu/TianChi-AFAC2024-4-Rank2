#!/bin/bash

python Type1-ChangShiCuoWu-BuWeiCuoWu-2Model.py -v final-test &&
python Type1-ChangShiCuoWu-ShiJianCuoWu-OnlyTime.py -v final-test &&
python Type2-ShuZhiDanWeiCuoWu.py -v final-test &&
python Type7-JiSuanCuoWu.py -v final-test &&
python Type8-YuJuChongFu-JiXie+LLM.py -v final-test &&
python postprocess.py -v final-test &&

echo "推理任务完成。"