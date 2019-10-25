#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from collections import Counter


class Tree(object):
    def __init__(self, train_data, target):
        """

        :param train_data: 训练集
        :param target:  训练标签
        """
        self.spliter = []
        self.train_data = train_data
        self.target = target
        self.fit(self.train_data, self.target)

    def entropy(self, elements):
        """
        算信息熵
        :param elements:
        :return:
        """
        counter = Counter(elements)
        probs = [counter[c] / len(elements) for c in elements]
        return - sum(p * np.log(p) for p in probs)

    def fit(self, train_data, target):
        """
        用训练集训练，会找到所有的合适切分点并记录到self.spliter里
        在最后的切割后训练集就是空了，无法继续切分，所以要忽略该报错
        :param train_data: 训练集
        :param target: 训练标签
        :return:
        """
        try:
            spliter = self.find_the_min_spilter(train_data, target)
            if spliter not in self.spliter and spliter:
                self.spliter.append(spliter)
                sub_train_data = train_data[train_data[spliter[0]] != spliter[1]]
                self.fit(sub_train_data, target)

        except TypeError as e:
            return

    def find_the_min_spilter(self, train_data, target):
        """
        寻找分割后最小熵的分割特征
        :param train_data: 训练集
        :param target: 训练标签
        :return:
        """
        x_fields = set(train_data.columns.tolist()) - {target}

        spliter = None
        min_entropy = float('inf')  # 预先设置成正无穷

        for f in x_fields:  # 循环所有特征名
            values = set(train_data[f])  # 取该特征所有数值做集合
            for v in values:  # 循环该特征所有数值集合
                sub_spliter_1 = train_data[train_data[f] == v][target].tolist()  # 计算等于该数值对目标特征分布
                entropy_1 = self.entropy(sub_spliter_1)  # 计算等于该数值时目标特征分布信息熵
                sub_spliter_2 = train_data[train_data[f] != v][target].tolist()  # 计算不等于该数值对目标特征分布
                entropy_2 = self.entropy(sub_spliter_2)  # 计算不等于该数值时目标特征分布信息熵
                entropy_v = entropy_1 + entropy_2  # 计算此数值对目标特征的总信息熵

                if entropy_v <= min_entropy:  # 如果信息熵小于当前最小信息熵则更新信息熵与特征
                    min_entropy = entropy_v
                    spliter = (f, v)

        return spliter

    def predicate(self, test_data):
        """
        模型用新数据进行预测
        :param test_data: 测试集
        :return: 测试标签
        """
        # 直接用self.train_data会影响该类后续预测
        tmp_data = self.train_data
        for k, v in self.spliter:
            # 如果测试集的字段值与self.spliter里字段有对应，则可以进行预测
            if test_data[k] == v:
                bought_list = tmp_data[tmp_data[k] == v]['bought'].tolist()
                return max(bought_list, key=bought_list.count)
            else:
                # 否则按分割流程进行分割数据集
                tmp_data = tmp_data[tmp_data[k] != v]

        return None

mock_data = {
        'gender': ['F', 'F', 'F', 'F', 'M', 'M', 'M'],
        'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],
        'family_number': [1, 1, 2, 1, 1, 1, 2],
        'bought': [1, 1, 1, 0, 0, 0, 1],
    }
train_data = pd.DataFrame.from_dict(mock_data)
target = 'bought'

t = Tree(train_data, target)

r = t.predicate({
        'gender': 'M',
        'income': '-10',
        'family_number': 1,
    })
print(r)

r = t.predicate({
        'gender': 'F',
        'income': '-10',
        'family_number': 1,
    })
print(r)

r = t.predicate({
        'gender': 'M',
        'income': '+10',
        'family_number': 1,
    })
print(r)
