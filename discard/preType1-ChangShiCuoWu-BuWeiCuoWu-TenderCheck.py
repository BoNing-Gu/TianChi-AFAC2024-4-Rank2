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
from utils import (read_docx_files, process_docx_files_2_para, process_docx_files_2_sents, process_docx_files_sents_2_longer, process_docx_files_year,
                   extract_context, clean_json_delimiters, process_docx_files_2_sents_WithNum, tender_document, tender_document_2_para, split_sentences)
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
    docx_files_dict_processed = process_docx_files_2_sents_WithNum(docx_files_dict)
    # 提取招标资格要求部分
    docx_files_dict_processed = tender_document(docx_files_dict_processed)
    # 转段落
    docx_files_dict_processed = tender_document_2_para(docx_files_dict_processed)

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
    output_csv_path = os.path.join(output1_dir, 'pre_type1-常识错误-不未错误-招标文件补充.csv')

    # 打开 CSV 文件，准备写入数据
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'result', 'para_id', 'sent_id'])  # 写入 CSV 文件头部

        # 抓取不未错误
        errs = ['不', '不为', '不会', '不能', '不得', '不具有', '会', '未', '无', '没有', '不可以', '可以', '免除',
                '被', '未被', '不被', '免除', '不合理', '合理', '未经', '无权', '有权', '存在', '不存在', '不用']
        errs_antonymy = ['', '为', '会', '可以', '可以', '具有', '不会', '', '有', '有', '可以', '不得', '不免除',
                         '未被', '被', '被', '不免除', '合理', '不合理', '经', '有权', '无权', '不存在', '存在', '须']
        for filename, doc in docx_files_dict_processed.items():
            print(f'处理文档：{filename}')
            for i, para in enumerate(doc):
                sentence_list = split_sentences(para)
                for j, sentence in enumerate(sentence_list):
                    found_keyword = False
                    word_list = jieba.lcut(sentence)
                    for k, word in enumerate(word_list):
                        for l, err in enumerate(errs):
                            if err == word:
                                found_keyword = True
                                possible_error_sent = sentence
                                print(f'原句：{possible_error_sent}')
                                temp_list = word_list[:]
                                temp_list[k] = errs_antonymy[l]
                                possible_error_sent_antonymy = ''.join(temp_list)
                                print(f'反义：{possible_error_sent_antonymy}')

                                prompt = (
                                        f"给定的段落：\n" +
                                        f"段落：{para}\n" +
                                        f"这个段落来自一篇招标公告的'投标人资格要求'部分，你需要判断以下这组逻辑词相互矛盾的句子中哪个不符合上下文语义：\n" +
                                        f"句子序号1：{possible_error_sent}\n" +
                                        f"句子序号2：{possible_error_sent_antonymy}\n" +
                                        f"请一步步分析，充分理解段落的属性（段落要求投标人存在某种行为或不存在某种行为），给出你的推理逻辑。\n" +
                                        f"示例：上文提到'承诺要求投标人应承诺近三年内未发生以下情况或失信行为：'，而句子表述为'没有被法院或其他国家行政管理部门判定为违法分包、转包、违规用工。'，由于上文要求的是'未发生以下失信行为'，该句子存在错误，应修正为'被法院或其他国家行政管理部门判定为违法分包、转包、违规用工。'\n\n" +
                                        """请按以下JSON格式提供回答：
                                        {\n
                                            "num": [
                                                "<你输出的有错误的句子序号>"
                                            ],\n
                                            "reason": [
                                                "<你的推理过程>"
                                            ]\n
                                        }
                                        警告：你的回复将直接用于javascript的JSON.parse解析，所以注意一定要以标准的JSON格式做回答，不要包含任何其他非JSON内容，否则你将被扣分！！！
                                        """
                                )
                                messages = [
                                    {"role": "system", "content": f"作为一位识别招标公告'投标人资格要求'部分的漏洞和矛盾的专家，你的任务是推断判断一组句子哪个真正符合上下文逻辑。请关注'不'、'没有'、'未'、'被'等逻辑词的错误，不必考虑句子中的其他潜在错误。"},
                                    {"role": "user", "content": prompt}
                                ]
                                response = client.chat.completions.create(
                                    model=args.model,
                                    messages=messages,
                                    stream=False,
                                    max_tokens=256,
                                    temperature=args.temp
                                )
                                # print(response.choices[0].message.content)
                                try:
                                    result = response.choices[0].message.content
                                    csv_writer.writerow([filename, result, i, j])
                                    answer.append([filename, result, i, j])
                                    print(f'输出：{filename}, {result}, {i}, {j}')
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
    output_excel_path = os.path.join(output2_dir, 'pre_type1-常识错误-不未错误-招标文件补充-备份.xlsx')
    answer_df = pd.DataFrame(answer, columns=['id', 'result', 'para_id', 'sent_id'])
    answer_df.to_excel(output_excel_path, index=False)

