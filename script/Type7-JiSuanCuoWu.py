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
                    default='glm-4-9b-chat',
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
    output_csv_path = os.path.join(output1_dir, 'answers_type7-计算错误.csv')
    # 打开 CSV 文件，准备写入数据
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'sent', 'possible_error_sent', 'sent_id'])  # 写入 CSV 文件头部

        # 抓取数值单位错误
        errs = ['元', '分钟', '个', '小时']
        for filename, doc in docx_files_dict_processed.items():
            print(f'处理文档：{filename}')
            if filename.startswith('平安'):
                print(f'跳过')
                continue
            for i, sentence in enumerate(doc):
                found_keyword = False
                for j, err in enumerate(errs):
                    if err in jieba.lcut(sentence):
                        found_keyword = True
                        possible_error_sent = sentence
                        print(f'原句：{possible_error_sent}')
                        break

                if not found_keyword:
                    continue

                # 提取上下文
                context_upper, context_lower = extract_context(i, doc, char_num)

                prompt = (
                        f"给定的上下文：" +
                        f"上文：{context_upper}\n" +
                        f"下文：{context_lower}\n" +
                        f"这段文本来自于研报、招标书或者法律条文，你需要判断以下这个包含数值信息的句子的数值是否计算正确，你可以尝试检查句子中是否存在数值计算关系，检查是否存在几个数值相加应该等于另一个但却不等于的情况：" +
                        f"句子：{possible_error_sent}\n" +
                        """
                        请综合上述信息，你给出的回复需要包含以下这两个字段：
                        1.TrueOrNot: 如果句子没有数值计算错误，字段填为True；如果句子有数值计算错误，字段填为False
                        2.sentence: 如果没有数值计算错误，该字段输出为空；如果有计算错误，输出这个句子中包含计算错误之处的最小粒度分句，请用 markdown 格式。
                        请按照以下JSON格式来回答：
                        {
                            "TrueOrNot": [
                                "<你判断的该句子的数值计算为正确或是错误>"
                            ],
                            "error_sentence": [
                                "<包含错误之处的最小粒度分句>"
                            ]
                        }
                        最后强调一下：你的回复将直接用于javascript的JSON.parse解析，所以注意一定要以标准的JSON格式做回答，不要包含任何其他非JSON内容，否则你将被扣分！！！
                        """
                )
                messages = [
                    {"role": "system",
                     "content": "作为一位识别金融文本中的漏洞和矛盾的专家，您的任务是对一个包含数值信息的句子进行判断，判断其句中含有的数值的计算关系是否正确，不存在数值计算关系的句子请不要考虑。"},
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
    output_excel_path = os.path.join(output2_dir, 'answers_type7-计算错误-备份.xlsx')
    answer_df = pd.DataFrame(answer, columns=['id', 'sent', 'possible_error_sent', 'sent_id'])
    answer_df = answer_df[['id', 'sent']]
    answer_df.to_excel(output_excel_path, index=False)