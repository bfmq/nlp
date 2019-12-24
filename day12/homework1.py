#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import jieba
import pandas as pd
from gensim.models import Word2Vec


stop_words = ['\n', '\r\n', '\r']
with open('../day5/data/stop/stopword.txt', encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.append(word.strip())


def cut(string):
    return [s for s in jieba.cut(string) if s not in stop_words]

data_source = "../day5/data/export_sql_1558435/sqlResult_1558435.csv"
data = pd.read_csv(data_source, encoding='utf-8')
data = data.fillna('')
news = data['content'].tolist()
news = news[800:900]
model = Word2Vec([cut(new) for new in news], size=50, window=5, min_count=1, workers=5)
model.save("./word2vec.model")

model = Word2Vec.load('./word2vec.model')


