import pandas as pd
import argparse

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

    result = pd.concat([type1_1, type1_2, type2, type7, type8])
    result.to_csv('../result_BuZhiDaoJiaoSha.csv', index=False)