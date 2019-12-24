#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import jieba
import pandas as pd
from sklearn.cluster import k_means
from gensim.models import Word2Vec


stop_words = ['\n', '\r\n', '\r']
with open('../day5/data/stop/stopword.txt', encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.append(word.strip())


def cut(string):
    return [s for s in jieba.cut(string) if s not in stop_words]


# 新闻读取，模型读取
data_source = "../day5/data/export_sql_1558435/sqlResult_1558435.csv"
data = pd.read_csv(data_source, encoding='utf-8')
data = data.fillna('')
news = data['content'].tolist()
news = news[800:900]
model = Word2Vec.load('./word2vec.model')

for i in range(len(news)):
    contents = [model[x] for x in cut(news[i])]
    key_words = set()

    # 为每个文章训练kmeans模型并获取中心点，中心点取3个
    k_means_model = k_means(contents, n_clusters=3)
    cluster_centers = k_means_model[0]

    for cluster_center in cluster_centers:
        # 循环中心点获取Word2Vec模型中与中心点向量最近的5个
        # 等于每篇文章15个关键字
        key_word = model.similar_by_vector(cluster_center, topn=5)
        for k, _ in key_word:
            key_words.add(k)

    print(f"第{i}文章的关键词是{key_words}")
