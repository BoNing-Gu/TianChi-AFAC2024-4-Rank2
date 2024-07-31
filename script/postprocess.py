import pandas as pd
import argparse
import json
import os

parser = argparse.ArgumentParser(
    description='PostProcess with Customizable Parameters')

parser.add_argument('-v',   # 版本
                    '--version',
                    type=str,
                    required=True,
                    help='Version')

args = parser.parse_args()

if __name__ == '__main__':
    version = args.version
    type1_1 = pd.read_excel(f'../answer_{version}-副本/answers_type1-常识错误-不未错误-备份.xlsx')
    type1_2 = pd.read_excel(f'../answer_{version}-副本/answers_type1-常识错误-时间错误-OnlyTime-备份.xlsx')
    type2 = pd.read_excel(f'../answer_{version}-副本/answers_type2-数值单位错误-备份.xlsx')
    type7 = pd.read_excel(f'../answer_{version}-副本/answers_type7-计算错误-备份.xlsx')
    type8 = pd.read_excel(f'../answer_{version}-副本/answers_type8-语句机械+大模型-备份.xlsx')

    result = pd.concat([type1_1, type1_2, type2, type7, type8]).drop_duplicates(subset=['id', 'sent'], inplace=False)
    result.to_csv('../result_BuZhiDaoJiaoSha.csv', index=False)

    json_path = r'../不知道叫啥_金融规则长文本_模板.json'
    final_json_path = r'../不知道叫啥_金融规则长文本_final.json'

    # 读取JSON模板文件并处理每一行
    with open(json_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            item = json.loads(line)  # 解析每一行的JSON对象
            id_value = item['id']
            # 查找对应id的数据
            selected_data = result[result['id'] == id_value]['sent'].tolist()
            for sent in selected_data:
                sent_list = []
                sent_list.append(sent)
                item['sents'].append(sent_list)
            if len(item['sents']) == 0:
                a_list = []
                a_list.append("年")
                item['sents'].append(a_list)
            data.append(item)

    with open(final_json_path, 'w', encoding='utf-8') as f:
        for index, item in enumerate(data):
            json.dump(item, f, ensure_ascii=False)
            if index < len(data) - 1:
                f.write('\n')  # 写入换行符，保证每行一个JSON对象

