import pandas as pd
import re
import jieba
jieba.load_userdict('dic.txt')

def compare_lists(list1, list2):
    same = []
    different = []
    for i, (x, y) in enumerate(zip(list1, list2)):
        if x == y:
            same.append(i)
        else:
            different.append(i)
    return same, different

def is_consecutive(nums):
    return set(nums) == set(range(min(nums), max(nums)+1))

def has_consecutive(nums):
    nums_set = set(nums)
    for num in nums_set:
        if num + 1 in nums_set:
            return True
    return False

def detect_duplicate_phrases(check_sentence, possible_error_sent, white_list):
    words = list(jieba.cut(check_sentence))
    words_copy = words[:].copy()
    empty_strings = [' '] * (len(words) - 1)
    words_copy += empty_strings
    result = []
    punctuation_symbols = ['%', '，', '。', '；', '、', '》', '《']

    for i in range(len(words) - 2):
        check_word = words_copy[-len(words):-1]
        same, different = compare_lists(words, check_word)
        duplicate_results = []
        duplicate_phrases = []
        if len(same) > 0:
            if has_consecutive(same):
                prev_j = None
                for j in range(len(words)):
                    if j in same:
                        if prev_j is None:
                            duplicate_phrases.append(words[j])
                            prev_j = j
                        if j - prev_j == 1:  # 连续
                            duplicate_phrases.append(words[j])
                            prev_j = j
                        if j - prev_j >= 2:  # 断开
                            strs = ''.join(duplicate_phrases)
                            if (len(duplicate_phrases) > 1) & (len(strs) > 6) & (strs in possible_error_sent) \
                                    & (any(strs.startswith(symbol) or strs.endswith(symbol) for symbol in punctuation_symbols)) \
                                    & (all(white_sent not in strs for white_sent in white_list)):
                                duplicate_results.append(strs)
                            duplicate_phrases = [words[j]]
                            prev_j = j
                strs = ''.join(duplicate_phrases)
                if (len(duplicate_phrases) > 1) & (len(strs) > 6) & (strs in possible_error_sent) \
                                    & (any(strs.startswith(symbol) or strs.endswith(symbol) for symbol in punctuation_symbols)) \
                                    & (all(white_sent not in strs for white_sent in white_list)):
                    duplicate_results.append(strs)
        # 更新
        words_copy.pop(-1)
        words_copy.insert(0, ' ')
        result.extend(duplicate_results)
    return result

def longest_string(strings):
    if not strings:
        return None  # 如果列表为空，返回 None
    longest = strings[0]
    for s in strings:
        if len(s) > len(longest):
            longest = s

    return longest

if __name__ == '__main__':
    # 示例句子
    sentence = "投标人应为中石油网内状态正常制造商，品名26030112压缩机组、26030114氢气压缩机、26030102石油气压缩机、26030114氢气压缩机之一正常状态"
    # sentence = "准入产品物资分类码：16010418、16012702、16019917、16011507、16011339、16011346、16010418、16012708、16011326、16011341、16012707、16011345、16011360、16011411、16020122、16020124、16020132、16020120、16019933）"
    # sentence = "评价需求，降低作业成本30%以上，实现探井市场占有率翻一番，装备综合水平达到国际先进，部分达到世界领先水平，支撑集团公司效益勘探和高质量发展。本次立项技术服务是依托外部强大技术力量和先进设计手段，联合开展井下仪器在多温度、压力场中机械结构及性能仿真技术研究，为仪器整体结构设计和关键部件选型、元器件测试及处理工艺优化进行理论指导，从源头提升仪器结构设计的可靠性和先进性，减少反复实验的浪费，降低作业成本30%以上。"
    # sentence = "合同签订后6-9个月内完成激发极化三维反演算法研究；合同签订后10-12个月内完成激发极化三维正反演软件开发及测试；合同签订后6-9个月内完成激发极化三维反演算法研究；"
    # sentence = "依法必须进行招标的项目的投标人有前款所列行为尚未构成犯罪的，处中标项目金额千分之五以上千分之十以下的罚款，处千分之五以上千分之十以下的罚款，对单位直接负责"
    # sentence = "依法必须进行招标的项目的投标人有前款所列行为尚未构成犯罪的，处中标项目金额千分之五以上千分之十以下的罚款，处千分之五以上千分之十以下的罚款，对单位直接负责的主管人员和其他直接责任人员处单位罚款数额百分之五以上百分之十以下的罚款；有违法所得的，并处没收违法所得；情节严重的，取消其一年至三年内参加依法必须进行招标的项目的投标资格并予以公告，直至由工商行政管理机关吊销营业执照。"

    # 检测重复部分并输出结果
    white_list = ['董事', '监事', '国有独资公司', '负债', '核算', '会计帐簿', '资产评估机构', '自治区', '自治县', '预算', '决算',
                  '是指各部门', '暂停供应商交易权限', '取消产品准入资格', '谈判小组', '安全生产许可证（许可范围须包含招标产品）'
                  '隐匿', '处罚的', '省', '民族乡', '带检验记录', '委托代理人', '试验', '附件',
                  '当周', 'http:', '价格方面', '价差方面', '.cnpcbidding.com', '2024年', '2024年05月', '4.1规定的', '.gov.cn）', ':00',
                  'U-key', '本保险合同', '审计', '单位', '刑事', '投标人', '招标中心', '登录', '网站', '项目经理', '警告', '中标', '招标']

    result = detect_duplicate_phrases(sentence, sentence, white_list)
    print(result)
