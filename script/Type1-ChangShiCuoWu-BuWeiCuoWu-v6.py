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
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('-m',   # 模型名
                    '--model',
                    type=str,
                    required=True,
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
    char_num = 200  # 上下文回顾窗口
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
    output_csv_path = os.path.join(output1_dir, 'answers_type1-常识错误-不未错误.csv')
    # 打开 CSV 文件，准备写入数据
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'sent', 'possible_error_sent', 'sent_id'])  # 写入 CSV 文件头部

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
                            possible_error_sent = sentence.replace(" ", "").replace(" ", "").replace("　", "")
                            print(f'原句：{possible_error_sent}')
                            temp_list = sentence_list[:]
                            temp_list[j] = errs_antonymy[k]
                            possible_error_sent_antonymy = ''.join(temp_list)
                            print(f'反义：{possible_error_sent_antonymy}')

                            # 提取上下文
                            context_upper, context_lower = extract_context(i, doc, char_num)
                            check_sentence = (context_upper + sentence + context_lower).replace(" ", "").replace(" ","").replace("　", "")

                            prompt = (
                                    f"你的任务是检查段落中是否存在逻辑词使用错误。逻辑词错误是指在句子中使用了不适当的反义词。" +
                                    f"段落：{check_sentence}\n" +
                                    f"请注意以下反义词对：\n"
                                    f"原词：{err}\n"
                                    f"反义词：{errs_antonymy[k]}\n\n"
                                    f"请判断原段落中的这个句子是否使用了不恰当的逻辑词。即，原本应该使用反义词中的一个词，但实际使用了另一种相对的反义词，从而导致语义上的错误。\n\n"
                                    f"句子：{possible_error_sent}\n\n"
                                    """
                                    请综合上述信息，你给出的回复需要包含以下这两个字段：\n
                                    1.TrueOrNot: 如果句子中的逻辑词使用正确，填写 `True`；如果使用错误，填写 `False`。\n
                                    2.error_sentence: 如果逻辑词使用正确，该字段为空；如果使用错误，输出包含错误的最小粒度分句，请用 markdown 格式标记。\n
                                    请按照以下JSON格式来回答：
                                    {
                                        "TrueOrNot": [
                                            "<你判断的该句子的逻辑词使用为正确或是错误>"
                                        ],
                                        "error_sentence": [
                                            "<包含错误之处的最小粒度分句>"
                                        ]
                                    }
                                    警告：你的回复将直接用于javascript的JSON.parse解析，所以注意一定要以标准的JSON格式做回答，不要包含任何其他非JSON内容，否则你将被扣分！！！
                                    """
                            )
                            messages = [
                                {"role": "system", "content": "作为一位识别金融类文本中逻辑词错误的专家，你的任务是判断句子中的逻辑词使用是否正确。"},
                                {"role": "user", "content": prompt}
                            ]
                            response = client.chat.completions.create(
                                model=args.model,
                                messages=messages,
                                stream=False,
                                max_tokens=4096,
                                temperature=args.temp
                            )
                            print(response.choices[0].message.content)
                            try:
                                parsed_json = json.loads(clean_json_delimiters(response.choices[0].message.content))
                                TrueOrNot = parsed_json['TrueOrNot'][0]
                                error_sentence = parsed_json['error_sentence'][0]
                                if TrueOrNot == 'Ture':  # 原句正确
                                    continue
                                if TrueOrNot == 'False':  # 原句错误
                                    csv_writer.writerow([filename, error_sentence, possible_error_sent, i])
                                    answer.append([filename, error_sentence, possible_error_sent, i])
                                    print(f'输出：{filename}, {error_sentence}, {possible_error_sent}, {i}')
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
    output_excel_path = os.path.join(output2_dir, 'answers_type1-常识错误-不未错误-备份.xlsx')
    answer_df = pd.DataFrame(answer, columns=['id', 'sent', 'possible_error_sent', 'sent_id'])
    answer_df.to_excel(output_excel_path, index=False)

