#!/usr/bin/env python
# -*- coding:utf8 -*-
# __author__ = '北方姆Q'
# __datetime__ = 2019/9/29 17:20


import os
import jieba
from collections import Counter
import pandas as pd


BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = f"{BaseDir}/day1/data/movie_comments.csv"
file_df = pd.read_csv(file_path)
df = file_df['comment']
df = df.dropna()
df = df.drop_duplicates()
df = df.str.strip()

FILE = ''
for i in df:
    if isinstance(i, str):
        FILE += i


def cut(string):
    return list(jieba.cut(string))


TOKENS = cut(FILE[:1000000])
words_count = Counter(TOKENS)
_2_gram_words = [
    TOKENS[i] + TOKENS[i+1] for i in range(len(TOKENS)-1)
]
_2_gram_word_counts = Counter(_2_gram_words)


def get_gram_count(word, wc):
    """
    获取字符串在总字符表中的次数
    :param word: 需要查询的字符串
    :param wc: 总字符表
    :return: 该字符串出现的次数，如没有则定为出现最少次数字符串的次数
    """
    return wc[word] if word in wc else wc.most_common()[-1][-1]


def two_gram_model(sentence):
    """
    分别计算句子中该单词在总字符表中出现的次数
    该单词跟后一单词在二连总字符表中出现的次数
    做比后的连续乘积
    :param sentence:  需要验证的句子
    :return:
    """
    tokens = cut(sentence)

    probability = 1

    for i in range(len(tokens)-1):
        word = tokens[i]
        next_word = tokens[i+1]

        _two_gram_c = get_gram_count(word + next_word, _2_gram_word_counts)
        _one_gram_c = get_gram_count(next_word, words_count)
        pro = _two_gram_c / _one_gram_c

        probability *= pro

    return probability


if "__main__" == __name__:
    r = two_gram_model("这个花特别好看")
    print(r)
    r = two_gram_model("花这个特别好看")
    print(r)
    r = two_gram_model("自然语言处理")
    print(r)
    r = two_gram_model("处语言理自然")
    print(r)
    r = two_gram_model("前天早上")
    print(r)
