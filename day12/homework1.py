#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-
# 这个是网上找的，根据自己的理解调整了一些并加了注释


import pandas as pd
import numpy as np
import jieba.posseg as pseg
from collections import defaultdict


class TextRank(object):
    def __init__(self, sentence, window=3, d=0.85, iternum=500):
        """

        :param sentence: 句子
        :param window: 滑动窗口数
        :param d: 阻尼系数
        :param iternum: 迭代次数
        """
        self.sentence = sentence
        self.window = window
        self.d = d
        self.iternum = iternum
        self.edge_dict = defaultdict()      # 节点之间的边字典

    def cutSentence(self):
        """
        分句
        :return:
        """
        tag_filter = ['a', 'd', 'n', 'v']
        self.word_list = [s.word for s in pseg.cut(self.sentence) if s.flag in tag_filter]

    def createNodes(self):
        """
        根据滑动窗口，建立节点，节点之间的边关系构建self.edge_dict
        :return:
        """
        for index, word in enumerate(self.word_list):
            if word not in self.edge_dict:
                tmp_set = set()
                # 滑动窗口左右确定
                left = index - self.window + 1
                right = index + self.window
                # 左右触发边界时调整
                left = 0 if left < 0 else left
                right = len(self.word_list) if right >= len(self.word_list) else right

                # 为i添加除了自己的窗口内其他节点
                for i in range(left, right):
                    if i != index: tmp_set.add(self.word_list[i])

                self.edge_dict[word] = tmp_set

    def createMatrix(self):
        """
        建立连接关系及初始权重
        :return:
        """
        self.matrix = np.zeros([len(set(self.word_list)), len(set(self.word_list))])
        self.word_index = {}#记录词的index
        self.index_dict = {}#记录节点index对应的词

        for i, v in enumerate(set(self.word_list)):
            self.word_index[v] = i
            self.index_dict[i] = v

        # 循环所有节点的相关节点，建立连接关系
        for key in self.edge_dict:
            for w in self.edge_dict[key]:      # 该节点的tmp_set
                self.matrix[self.word_index[key]][self.word_index[w]] = 1
                self.matrix[self.word_index[w]][self.word_index[key]] = 1

        # 计算该节点所有的连接数，求平均
        for j in range(self.matrix.shape[1]):
            all_sum = 0
            for i in range(self.matrix.shape[0]):
                all_sum += self.matrix[i][j]
            for i in range(self.matrix.shape[0]):
                self.matrix[i][j] /= all_sum

    def calPR(self):
        """
        根据textrank公式迭代权重值
        :return:
        """
        self.PR = np.ones([len(set(self.word_list)), 1])
        for _ in range(self.iternum):
            self.PR = (1 - self.d) + self.d * np.dot(self.matrix, self.PR)

    def getResult(self):
        """
        获取词权重排序
        :return:
        """
        word_pr = {}
        for i in range(len(self.PR)):
            word_pr[self.index_dict[i]] = self.PR[i][0]
        return sorted(word_pr.items(), key=lambda x: x[1], reverse=True)


data_source = "../day5/data/export_sql_1558435/sqlResult_1558435.csv"
data = pd.read_csv(data_source, encoding='utf-8')
data = data.fillna('')
news = data["content"][0:100]
for new in news:
    tr = TextRank(new)
    tr.cutSentence()
    tr.createNodes()
    tr.createMatrix()
    tr.calPR()
    r = tr.getResult()
    print(r)
