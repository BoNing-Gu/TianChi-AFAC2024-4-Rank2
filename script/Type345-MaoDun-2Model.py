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
    output_csv_path = os.path.join(output1_dir, 'answers_type345-矛盾.csv')

    # 读取Qwen
    qwen_csv_path = os.path.join(output1_dir, 'pre_type345-矛盾.csv')
    qwen = pd.read_csv(qwen_csv_path)
    # 打开 CSV 文件，准备写入数据
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'sent', 'para_id'])  # 写入 CSV 文件头部

        # 抓取矛盾
        for filename, doc in docx_files_dict_processed.items():
            print(f'处理文档：{filename}')
            if filename.endswith('法') or filename.startswith('平安'):
                print(f'跳过')
                continue
            for i, para in enumerate(doc):
                # 提取qwen答案
                try:
                    qwen_answer = qwen[(qwen['id'] == filename) & (qwen['para_id'] == i)]['result'].values[0]
                except Exception as e:
                    qwen_answer = ""
                    print(f"qwen无回答: {e}")
                possible_error_para = para
                prompt = (
                        f"以下段落可能包含人为设置的矛盾表述。你的任务是判断这个段落中是否存在真正的矛盾，并指出矛盾的具体内容。\n" +
                        f"段落：{possible_error_para}\n" +
                        f"段落中可能存在以下三种矛盾类型：\n" +
                        f"a. 逻辑矛盾：如果两个句子的逻辑推理相互排斥，且在任何合理解释下都无法兼容，则认为存在逻辑矛盾。\n" +
                        f"b. 时间矛盾：如果两个句子中的时间信息相互冲突，即无法在同一时间框架下成立，则认为存在时间矛盾。\n" +
                        f"c. 数值矛盾：如果两个句子中的数值信息相互排斥，即在任何合理计算下都无法同时成立，则认为存在数值矛盾。\n" +
                        f"你的金融助手对于句子可能涉及的矛盾给出了以下建议，你可以综合你的知识和助手的建议进行判断：" +
                        f"{qwen_answer}\n" +
                        f"请注意，如果段落中的条款看似不同但并不真正矛盾（例如，法律条款可能涉及不同的情境），则不应视为矛盾。\n\n" +
                        """
                        请综合上述信息，你给出的回复需要包含以下这三个字段：
                        1.TrueOrNot: 如果段落中不存在真正的矛盾表述，填写 `True`；如果存在真正的矛盾表述，填写 `False`。\n
                        2.sentence: 如果存在矛盾，指出与后文存在矛盾的句子，使用 markdown 格式输出该句子中包含矛盾部分的最小粒度分句。\n
                        3.contradiction: 找到与 `sentence` 存在矛盾的句子，使用 markdown 格式输出该句子中包含矛盾部分的最小粒度分句。\n\n
                        请按照以下JSON格式来回答：
                        {
                            "TrueOrNot": [
                                "<你判断的段落表述是否正确><True 或 False>"
                            ],
                            "sentence": [
                                "<包含矛盾部分的最小粒度分句或空字符串>"
                            ],
                            "contradiction": [
                                "<与 sentence 存在矛盾的最小粒度分句或空字符串>"
                            ]
                        }
                        注意：如果你认为段落中不存在真正的矛盾，sentence和contradiction应返回空字符串。
                        最后强调一下：你的回复将直接用于javascript的JSON.parse解析，所以注意一定要以标准的JSON格式做回答，不要包含任何其他非JSON内容，否则你将被扣分！！！
                        """
                )
                messages = [
                    {"role": "system",
                     "content": "作为一位识别金融文本中的漏洞和矛盾的专家，你的任务是判断段落是否含有真正的矛盾表述。请确保在判断时考虑上下文的合理性和一致性。只有明确的矛盾才会被认为是有效矛盾，不包括模糊的表达。"},
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
                    sentence = parsed_json['sentence'][0]
                    contradiction = parsed_json['contradiction'][0]
                    if TrueOrNot == 'Ture':  # 不存在矛盾
                        continue
                    if TrueOrNot == 'False':  # 存在矛盾
                        csv_writer.writerow([filename, [sentence, contradiction], i])
                        answer.append([filename, [sentence, contradiction], i])
                        print(f'输出：{filename}, [{sentence}, {contradiction}, {i}]')
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
    output_excel_path = os.path.join(output2_dir, 'answers_type345-矛盾-备份.xlsx')
    answer_df = pd.DataFrame(answer, columns=['id', 'sent', 'para_id'])
    answer_df.to_excel(output_excel_path, index=False)