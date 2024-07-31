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
    # # 更长分句
    # docx_files_dict_processed = process_docx_files_sents_2_longer(docx_files_dict_processed, 3)

    # 设置
    white_list = ['董事', '监事', '国有独资公司', '负债', '核算', '会计帐簿', '资产评估机构', '自治区', '自治县', '预算', '决算', '法', '收入', '人民',
                  '是指各部门', '暂停供应商交易权限', '取消产品准入资格', '谈判小组', '安全生产许可证（许可范围须包含招标产品）',
                  '隐匿', '处罚的', '省', '民族乡', '带检验记录', '委托代理人', '试验', '附件', '保险', '证明', '政府', '报告',
                  '当周', 'http:', '价格方面', '价差方面', '.com', '2024年', '4.1', '.gov.cn', ':00',
                  'U-key', '本保险合同', '审计', '单位', '刑事', '投标人', '招标中心', '登录', '网站', '项目经理', '警告', '中标', '招标']
    char_num = 30
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
    output_csv_path = os.path.join(output1_dir, 'answers_type8-语句机械+大模型.csv')
    # 打开 CSV 文件，准备写入数据
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'sent', 'possible_error_sent', 'sent_id'])  # 写入 CSV 文件头部

        # 抓取语句重复
        for filename, doc in docx_files_dict_processed.items():
            print(f'处理文档：{filename}')
            if filename.startswith('平安'):
                print(f'跳过')
                continue
            for i, sentence in enumerate(doc):
                if i < 6:
                    continue
                context_upper, context_lower = extract_context(i, doc, char_num)
                possible_error_sent = sentence.replace(" ", "").replace(" ", "").replace("　", "")
                check_sentence = (context_upper + sentence + context_lower).replace(" ", "").replace(" ", "").replace("　", "")
                result = detect_duplicate_phrases(check_sentence, possible_error_sent, white_list)
                output = longest_string(result)
                if output is not None:
                    prompt = (
                            f"这段文本可能包含人为插入的重复语句作为陷阱，请判断这些重复是否多余。以下是段落和可能的重复部分：" +
                            f"段落：{check_sentence}\n" +
                            f"重复部分：{output}\n" +
                            "请根据以下标准判断：\n" +
                            "- 如果重复部分对理解内容有帮助，且在段落中有其必要性，则不是重复陷阱；\n" +
                            "- 如果重复部分显得突兀且不必要，则可能是重复陷阱。\n\n" +
                            """
                            请综合上述信息，你给出的回复需要包含以下这两个字段：\n
                            1.TrueOrNot: 如果段落没有重复陷阱，填写 `True`；如果有重复陷阱，填写 `False`。\n
                            2.error_sentence: 如果没有重复陷阱，该字段输出为空；如果有重复陷阱，输出这个句子中包含重复处的最小粒度分句，请用 markdown 格式。\n
                            请按照以下JSON格式来回答：
                            {
                                "TrueOrNot": [
                                    "<你判断的该句子的重复部分为正确或是错误>"
                                ],
                                "error_sentence": [
                                    "<包含重复陷阱之处的最小粒度分句>"
                                ]
                            }
                            最后强调一下：你的回复将直接用于javascript的JSON.parse解析，所以注意一定要以标准的JSON格式做回答，不要包含任何其他非JSON内容，否则你将被扣分！！！
                            """
                    )
                    messages = [
                        {"role": "system",
                         "content": "作为一位识别金融文本中的漏洞和矛盾的专家，你的任务是判断句子中的重复部分是否为陷阱。"},
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
    output_excel_path = os.path.join(output2_dir, 'answers_type8-语句机械+大模型-备份.xlsx')
    answer_df = pd.DataFrame(answer, columns=['id', 'sent', 'possible_error_sent', 'sent_id'])
    answer_df = answer_df[['id', 'sent']]
    answer_df.to_excel(output_excel_path, index=False)