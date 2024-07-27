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
    output_csv_path = os.path.join(output1_dir, 'pre_type1-常识错误-不未错误.csv')
    # 打开 CSV 文件，准备写入数据
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'judge', 'Reason', 'sent_id'])  # 写入 CSV 文件头部

        # 抓取不未错误
        errs = ['不', '不为', '不会', '不能', '不得', '不具有', '会', '未', '无', '没有', '不可以', '可以', '免除',
                '被', '未被', '具备', '不具备', '免除', '不合理', '合理', '未经过', '无权', '有权', '存在']
        errs_antonymy = ['', '为', '会', '可以', '可以', '具有', '不会', '', '有', '有', '可以', '不得', '不免除',
                         '未被', '被', '不具备', '具备', '不免除', '合理', '不合理', '经过', '有权', '无权', '不存在']
        for filename, doc in docx_files_dict_processed.items():
            print(f'处理文档：{filename}')
            for i, sentence in enumerate(doc):
                found_keyword = False
                sentence_list = jieba.lcut(sentence)
                for j, word in enumerate(sentence_list):
                    for k, err in enumerate(errs):
                        if err == word:
                            found_keyword = True
                            possible_error_sent = sentence
                            print(f'原句：{possible_error_sent}')
                            temp_list = sentence_list[:]
                            temp_list[j] = errs_antonymy[k]
                            possible_error_sent_antonymy = ''.join(temp_list)
                            print(f'反义：{possible_error_sent_antonymy}')

                            prompt = (
                                    f"这段文本来自于研报、招标书或者法律条文，这句话是否存在逻辑词使用的错误？" +
                                    f"句子：{possible_error_sent}\n" +
                                    f"可能出错的逻辑词：{err}\n" +
                                    """
                                    请综合上述信息，你给出的回复需要包含以下这两个字段：
                                    1.judge: 如果句子中的逻辑词没有错误，填写`True`；如果句子中的逻辑词有错误，填写`False`。
                                    2.reason: 一步步思考，按要点给出你做出判断的推理逻辑。
                                    请按照以下JSON格式来回答：
                                    {
                                        "judge": [
                                            "<句子是正确还是错误>"
                                        ],
                                        "reason": [
                                            "<你的推理逻辑>"
                                        ]
                                    }
                                    警告：你的回复将直接用于javascript的JSON.parse解析，所以注意一定要以标准的JSON格式做回答，不要包含任何其他非JSON内容，否则你将被扣分！！！
                                    """
                            )
                            messages = [
                                {"role": "system", "content": "作为一位识别金融文本中的漏洞和矛盾的专家，您的任务是判断一个含有逻辑词的句子是否正确，并给出你的推理逻辑"},
                                {"role": "user", "content": prompt}
                            ]
                            response = client.chat.completions.create(
                                model=args.model,
                                messages=messages,
                                stream=False,
                                max_tokens=1024,
                                temperature=args.temp
                            )
                            print(response.choices[0].message.content)
                            try:
                                parsed_json = json.loads(clean_json_delimiters(response.choices[0].message.content))
                                judge = parsed_json['judge'][0]
                                reason = parsed_json['reason'][0]
                                csv_writer.writerow([filename, judge, reason, i])
                                answer.append([filename, judge, reason, i])
                                print(f'输出：{filename}, {judge}, {reason}, {i}')
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
    output_excel_path = os.path.join(output2_dir, 'pre_type1-常识错误-不未错误-备份.xlsx')
    answer_df = pd.DataFrame(answer, columns=['id', 'judge', 'Reason', 'sent_id'])
    answer_df.to_excel(output_excel_path, index=False)

