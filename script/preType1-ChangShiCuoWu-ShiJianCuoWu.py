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
    # 分句
    docx_files_dict_processed = process_docx_files_2_sents(docx_files_dict)
    # 提取年份
    docx_files_dict_processed = process_docx_files_year(docx_files_dict_processed)

    # 设置
    char_num = 200  # 上下午回顾窗口
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
    output_csv_path = os.path.join(output1_dir, 'pre_type1-常识错误-不未错误.csv')

    # 打开 CSV 文件，准备写入数据
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'result', 'sent_id'])  # 写入 CSV 文件头部

        # 抓取时间错误+时间数值缺失
        errs = ['年', '月', '日', '上午', '下午', '日期']
        for filename, doc in docx_files_dict_processed.items():
            print(f'处理文档：{filename}')
            if filename.startswith('平安'):
                print(f'跳过')
                continue
            for i, sentence in enumerate(doc):
                if i == 0:  # 跳过基本年份信息
                    continue
                found_keyword = False
                for j, err in enumerate(errs):
                    if sentence.find(err) != -1:
                        found_keyword = True
                        possible_error_sent = sentence
                        print(f'原句：{possible_error_sent}')
                        break

                if not found_keyword:
                    continue

                # 提取上下文
                context_upper, context_lower = extract_context(i, doc, char_num)

                prompt = (
                        "这段文本来源于研报、招标书或法律条文：\n" +
                        f"这篇文档中关于文档基本年份信息的句子为：{doc[0]}\n" +
                        f"上文内容：{context_upper}\n" +
                        f"请检查下面的句子，判断其中是否存在时间信息的使用错误，并结合上文内容提供相关常识或金融知识以辅助判断。\n" +
                        f"待检查句子：{possible_error_sent}\n" +
                        f"请逐步分析句子涉及的时间信息，并在句子时间信息表述与常识或上文不符的情况下，提供修正后的金融知识以帮助我理解这个错误。\n" +
                        "请按以下JSON格式提供回答：\n"
                        "[\n"
                        "    \"<待检查句子修正后的金融知识>\"\n"
                        "]"
                )
                messages = [
                    {"role": "system", "content": "作为金融文本分析助手，你的任务是提供金融相关知识以帮助判断一个句子的时间信息是否符合现实和上下文。"},
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
    output_excel_path = os.path.join(output2_dir, 'pre_type1-常识错误-时间错误-备份.xlsx')
    answer_df = pd.DataFrame(answer, columns=['id', 'result', 'sent_id'])
    answer_df.to_excel(output_excel_path, index=False)