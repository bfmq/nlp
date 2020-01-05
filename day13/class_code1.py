#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import random
from scipy.spatial.distance import cosine
from functools import reduce
from operator import and_

fpath = '../day5/data/export_sql_1558435/sqlResult_1558435.csv'
data = pd.read_csv(fpath, encoding='utf-8')
news_content = data['content'].tolist()


def token(string):
    return re.findall(r'[\d|\w]+', string)


def cut(string):
    return ' '.join(jieba.cut(string))

news_content = [token(str(n)) for n in news_content]
news_content = [' '.join(n) for n in news_content]
news_content = [cut(n) for n in news_content]
sample_num = 50000
sub_sample_news = news_content[:sample_num]
vectorized = TfidfVectorizer(max_features=10000)
X = vectorized.fit_transform(sub_sample_news)
X_array = X.toarray()
# print(vectorized.vocabulary_)
document_id_1, document_id_2, document_id_3 = random.randint(0, 1000), random.randint(0, 1000), random.randint(0,1000)
vector_of_d_1 = X[document_id_1].toarray()[0]
vector_of_d_2 = X[document_id_2].toarray()[0]
vector_of_d_3 = X[document_id_3].toarray()[0]


def get_distance(v1, v2):
    return cosine(v1, v2)


r = get_distance(vector_of_d_1, vector_of_d_2)
print(r)
r = get_distance(vector_of_d_1, vector_of_d_3)
print(r)


def distance_with_document_1(i):
    return get_distance(vector_of_d_1, X[i].toarray()[0])

r = sorted(list(range(10000)), key=distance_with_document_1)
print(r)


def naive_search(keywords):
    """
    暴力搜索，循环查找关键字与每篇文章每个字符的关系
    :param keywords:
    :return:
    """
    news_ids = [i for i, n in enumerate(news_content) if all(w in n for w in keywords)]
    return news_ids

ids = naive_search('美军 司令 航母'.split())


word2id = vectorized.vocabulary_
id2word = {d: w for w, d in word2id.items()}
transposed_x = X.transpose().toarray()


def search_engine(query):
    """
    布尔搜索，获取关键字的id后查找包含的文档
    排序返回
    :param query:
    :return:
    """
    words = query.split()

    query_vec = vectorized.transform([' '.join(words)]).toarray()[0]

    candidate_ids = [word2id[w] for w in words]

    documents_ids = [
        set(np.where(transposed_x[_id])[0]) for _id in candidate_ids
    ]

    merged_documents = reduce(and_, documents_ids)

    sorted_documents_id = sorted(merged_documents, key=lambda i: get_distance(query_vec, X[i].toarray()))

    return sorted_documents_id

r = search_engine('美军 司令 航母')
print(r)
