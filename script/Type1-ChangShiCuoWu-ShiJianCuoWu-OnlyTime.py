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
                   process_docx_files_year, extract_context, clean_json_delimiters, process_docx_all_time)
from check_duplicate_in_sent import detect_duplicate_phrases, longest_string

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('-m',   # 模型名
                    '--model',
                    type=str,
                    default=r'glm-4-9b-chat',
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
                    default=0.1,
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
    # 提取时间信息
    docx_files_dict_processed = process_docx_all_time(docx_files_dict_processed)

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
    output_csv_path = os.path.join(output1_dir, 'answers_type1-常识错误-时间错误-OnlyTime.csv')

    # 打开 CSV 文件，准备写入数据
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'sent'])  # 写入 CSV 文件头部

        for filename, doc in docx_files_dict_processed.items():
            print(f'处理文档：{filename}')
            prompt = (
                f"以下句子来自同一篇文档，每个句子都包含至少一处时间信息，请检查是否句子存在时间信息上的错误，例如时间与常识不符、时间前后颠倒等。\n"
                f"句子列表：\n"
            )
            for sentence in doc:
                prompt += f"- {sentence}\n"
            prompt += (
                """
                请综合上述信息，你给出的回复需要包含以下这个字段：\n
                1.error_sentence: 如果所有句子都没有时间错误，这个字段留空；如果某个句子有时间错误，输出包含错误时间的最小粒度分句；如果有多个句子存在错误，则将错误之处都写入列表中。
                请按照以下JSON格式来回答：\n
                {
                    "error_sentence": [
                        "<原句中包含错误之处的最小粒度分句>", "<原句中包含错误之处的最小粒度分句>", ...
                    ]
                }\n
                最后强调一下：你的回复将直接用于javascript的JSON.parse解析，所以注意一定要以标准的JSON格式做回答，不要包含任何其他非JSON内容，否则你将被扣分！！！\n
                """
            )

            messages = [
                {"role": "system", "content": "作为一位识别金融文本中的漏洞和矛盾的专家，您的任务是判断一组包含时间信息的句子是否存在错误，如有错误需指出错误之处。"},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=False,
                max_tokens=2048,
                temperature=args.temp
            )
            print(response.choices[0].message.content)
            try:
                parsed_json = json.loads(clean_json_delimiters(response.choices[0].message.content))
                error_sentence = parsed_json['error_sentence']
                if len(error_sentence) != 0:
                    for sentence in error_sentence:
                        csv_writer.writerow([filename, sentence])
                        answer.append([filename, sentence])
                        print(f'输出：{filename}, {sentence}')
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
    output_excel_path = os.path.join(output2_dir, 'answers_type1-常识错误-时间错误-OnlyTime-备份.xlsx')
    answer_df = pd.DataFrame(answer, columns=['id', 'sent'])
    white_list_id = ['审计法', '保险', '化工行业周报', '平安', '玉门油田', '电力', '西南']
    white_list_in = ['招标文件', '5个工作日', '天然气', '投标文件递交', '估算金额', '网上开标', '招标文件的获取',
                     '票房', '五一', 'Q1', '编号', '招标公告', '有效期', '货代', '预算管理', '电子', '央行', '23:59']
    answer_df = answer_df[~answer_df['id'].str.contains('|'.join(white_list_id))]
    answer_df = answer_df[~answer_df['sent'].str.contains('|'.join(white_list_in))]
    answer_df = answer_df[['id', 'sent']]
    answer_df.to_excel(output_excel_path, index=False)