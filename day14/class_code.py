#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import math
import jieba


text = [
    "自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。",
    "它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。",
    "自然语言处理是一门融语言学、计算机科学、数学于一体的科学。",
    "因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，所以它与语言学的研究有着密切的联系，但又有重要的区别。",
    "自然语言处理并不是一般地研究自然语言，而在于研制能有效地实现自然语言通信的计算机系统，特别是其中的软件系统。因而它是计算机科学的一部分。"
]


class BM25:
    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc) for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 文档中词频
        self.df = {}  # 词 文档数
        self.idf = {}  # idf数值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 词 ： 词频
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1  # 词：文档数量
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1) /
                      (self.f[index][word] + self.k1 * (1 - self.b + self.b * d / self.avgdl)))
        return score

    def similar(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores


doc = []
for sent in text:
    words = list(jieba.cut(sent))
    doc.append(words)

s = BM25(doc)

print(s.f)
print(s.idf)

test = "自然语言处理属于计算机科学领域里的人工智能领域"
print(s.similar(list(jieba.cut(test))))
