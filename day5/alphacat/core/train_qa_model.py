#!/usr/bin/env python
# coding=utf-8

import jieba
import pandas as pd
import multiprocessing
from gensim.models import Word2Vec


# 加载停用词表
stop_words = ['\n', '\r\n', '\r']
with open('../../data/stop/stopword.txt', encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.append(word.strip())


def content_process(content):
    """
    切文本
    :param content:
    :return:
    """
    try:
        return [c for c in list(jieba.cut(content)) if c not in stop_words]

    except Exception as e:
        return []

df = pd.read_csv('./qa_corpus.csv', encoding='utf-8')
questions = [content_process(q) for q in df['question']]
print(questions)

model = Word2Vec(questions, size=50, sg=1, window=10, workers=multiprocessing.cpu_count())
model.save('./qa50.model')
model.wv.save_word2vec_format('./qa50.model.wv.vectors.npy')

model = Word2Vec(questions, size=100, sg=1, window=10, workers=multiprocessing.cpu_count())
model.save('./qa100.model')
model.wv.save_word2vec_format('./qa100.model.wv.vectors.npy')
