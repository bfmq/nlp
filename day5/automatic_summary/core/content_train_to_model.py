#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import jieba
import pandas as pd
from gensim.models import word2vec
from threading import Thread


# 加载停用词
with open('../data/stop/stopword.txt', 'r', encoding='utf-8') as f:
    STOP_WORDS = [line.strip() for line in f.readlines()]


file_path = "../data/export_sql_1558435/"
new_file_name = 'content.csv'

df = pd.read_csv(file_path+new_file_name, header=None, names=['id', 'content'])


def train(model, data):
    for content in data:
        train_list = []

        if isinstance(content, str):        # 必须判断为字符串类型，否则模型训练会报错！
            tokens = list(jieba.cut(content))   # 分词
            for token in tokens:                # 去掉停用词
                if token.strip() and token.strip() not in STOP_WORDS:
                    train_list.append(token.strip())

        model.build_vocab([train_list], update=True)
        model.train([train_list], total_examples=model.corpus_count, epochs=1)


# 74447

for i in range(8, 25):
    model = word2vec.Word2Vec.load("../models/wiki_and_content/wiki_and_content.model")
    data = df['content'][i*3000:(i+1)*3000]
    train(model, data)
    model.save('../models/wiki_and_content/wiki_and_content.model')
    print(i)
