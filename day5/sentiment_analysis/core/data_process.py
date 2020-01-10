#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import pandas as pd
from gensim.models.word2vec import Word2Vec


# random.seed = 16
train = pd.read_csv('../../data/ai_challenger_sentiment_analysis/train/trainingset.csv')
validation = pd.read_csv("../../data/ai_challenger_sentiment_analysis/validationset.csv")
test = pd.read_csv("../../data/ai_challenger_sentiment_analysis/testa.csv")

# 加载停用词表
stop_words = ['\n', '\r\n', '\r']
with open('../../data/stop/stopword.txt', encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.append(word.strip())


def content_process(content):
    return [c for c in content if c not in stop_words]


train.content = train.content.map(lambda x: content_process(x))
validation.content = validation.content.map(lambda x: content_process(x))
test.content = test.content.map(lambda x: content_process(x))
train.to_csv("../../data/ai_challenger_sentiment_analysis/train_char.csv", index=None)
validation.to_csv("../../data/ai_challenger_sentiment_analysis/validation_char.csv", index=None)
test.to_csv("../../data/ai_challenger_sentiment_analysis/test_char.csv", index=None)


line_sent = []
for s in train["content"]:
    line_sent.append(s)

word2vec_model = Word2Vec(line_sent, size=100, window=10, min_count=1, workers=4, iter=15)
word2vec_model.wv.save_word2vec_format("../../models/ai_challenger_sentiment_analysis/chars.vector", binary=True)
