import os
import torch
import time
import json
import csv
import shutil
import pandas as pd
import argparse
import jieba
jieba.load_userdict('dic.txt')
from openai import OpenAI
from utils import (read_docx_files, process_docx_files_2_para, process_docx_files_2_sents, process_docx_files_sents_2_longer,
                   process_docx_files_year, extract_context, clean_json_delimiters)
from check_duplicate_in_sent import detect_duplicate_phrases, longest_string

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8001/v1',
                    help='Model URL')
parser.add_argument('-m',   # 模型名
                    '--model',
                    type=str,
                    default='Tongyi-Finance-14B-Chat-Int4',
                    help='Model name for the chatbot')
parser.add_argument('-v',   # 版本
                    '--version',
                    type=str,
                    required=True,
                    help='Version')
parser.add_argument('-d',   # 数据目录路径
                    '--data',
                    type=str,
                    default=r'../data_B',
                    help='Data Dir')
parser.add_argument('--temp',
                    type=float,
                    default=0.3,
                    help='Temperature for text generation')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

if __name__ == "__main__":
    # 读取处理后的文档
    docx_files_dict = read_docx_files(args.data)
    # 分段
    docx_files_dict_processed = process_docx_files_2_para(docx_files_dict)

    # 设置
    openai_api_key = "EMPTY"
    openai_api_base = args.model_url
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    answer = []
    version = args.version
    output1_dir = os.path.join('..', f'answer_{version}')
    if not os.path.exists(output1_dir):
        os.makedirs(output1_dir)
    output_csv_path = os.path.join(output1_dir, 'pre_type345-矛盾.csv')
    # 打开 CSV 文件，准备写入数据
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'result', 'para_id'])  # 写入 CSV 文件头部

        # 抓取矛盾
        for filename, doc in docx_files_dict_processed.items():
            print(f'处理文档：{filename}')
            if filename.endswith('法') or filename.startswith('平安'):
                print(f'跳过')
                continue
            for i, para in enumerate(doc):
                possible_error_para = para
                prompt = (
                        f"请理解以下段落内容，为我概述这个段落相关的金融知识。\n" +
                        f"段落：{possible_error_para}\n" +
                        f"请一步步思考，向我简明地概括段落涉及的金融知识。如果段落表述存在矛盾，就为我提供改正后的金融知识。"
                )
                messages = [
                    {"role": "system",
                     "content": "作为分析金融文本的助手，你专注于为我提供金融相关知识，辅助我判断一个段落是否正确。请确保在回答时考虑上下文的合理性和一致性。"},
                    {"role": "user", "content": prompt}
                ]
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=False,
                    max_tokens=256,
                    temperature=args.temp
                )
                try:
                    result = response.choices[0].message.content
                    csv_writer.writerow([filename, result, i])
                    answer.append([filename, result, i])
                    print(f'输出：{filename}, {result}, {i}')
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败: {e}")
                except IndexError as e:
                    print(f"索引越界: {e}")
                except ValueError as e:
                    print(f"值错误：{e}")
                except TypeError as e:
                    print(f"类型错误：{e}")

    output2_dir = os.path.join('..', f'answer_{version}-副本')
    if not os.path.exists(output2_dir):
        os.makedirs(output2_dir)
    output_excel_path = os.path.join(output2_dir, 'pre_type345-矛盾-备份.xlsx')
    answer_df = pd.DataFrame(answer, columns=['id', 'result', 'para_id'])
    answer_df.to_excel(output_excel_path, index=False)