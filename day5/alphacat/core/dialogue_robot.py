#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import json
import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models import word2vec
from sklearn.externals import joblib


# word_frequency = json.load(open('../../models/qa/word_frequency.json', encoding='utf-8'))
# model = word2vec.Word2Vec.load("../../models/qa/qa50.model")
# kmeans = joblib.load('../../models/qa/kmeans_50_7.pkl')
# bool_json = json.load(open(f'../../models/qa/bool_50_7.json', encoding='utf-8'))
# sentence_vecs = json.load(open(f'../../models/qa/sentence_vecs50.json', encoding='utf-8'))
#
# df = pd.read_csv('../../data/qa/qa_corpus.csv', encoding='utf-8')
# questions, answers = df['question'], df['answer']
# # 加载停用词表
# stop_words = ['\n', '\r\n', '\r']
# with open('../../data/stop/stopword.txt', encoding='utf-8') as f:
#     for word in f.readlines():
#         stop_words.append(word.strip())

word_frequency = json.load(open('models/qa/word_frequency.json', encoding='utf-8'))
model = word2vec.Word2Vec.load("models/qa/qa50.model")
kmeans = joblib.load('models/qa/kmeans_50_5.pkl')
bool_json = json.load(open(f'models/qa/bool_50_5.json', encoding='utf-8'))
sentence_vecs = json.load(open(f'models/qa/sentence_vecs50.json', encoding='utf-8'))

df = pd.read_csv('data/qa/qa_corpus.csv', encoding='utf-8')
questions, answers = df['question'], df['answer']
# 加载停用词表
stop_words = ['\n', '\r\n', '\r']
with open('data/stop/stopword.txt', encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.append(word.strip())


def content_process(content):
    try:
        return [c for c in list(jieba.cut(content)) if c not in stop_words]
    except Exception as e:
        return []


def cut(text):
    return ' '.join(jieba.cut(text))


class SifEmbedding(object):
    def __init__(self, text):
        """

        :param text: 正文本
        """
        self.text = text
        self.alpha = 1e-4
        self.sentences_vec = self.get_weight_average(self.text)

    def get_weight_average(self, text):
        """
        SIF公式第一步
        :param text:
        :return:
        """
        max_fre = max(word_frequency.values())
        sen_vec = np.zeros_like(model.wv['银行'])

        words = cut(text).split()
        words = [w for w in words if w in model]

        # SIF公式第一步，（学习率/比重+学习率）* 词向量的加和
        for w in words:
            fre = word_frequency.get(w, max_fre)
            weight = self.alpha / (fre + self.alpha)
            sen_vec += weight * model.wv[w]

        if len(words) != 0:         # 这里必须要这么写，不能写三元运算
            sen_vec /= len(words)
        return sen_vec

    def remove_pca(self, vec, npc=1):
        u = self.compute_pca(vec, npc)
        if npc == 1:
            vec -= vec.dot(u.T) * u
        else:
            vec -= vec.dot(u.T).dot(u)
        return vec

    def compute_pca(self, vec, npc=1):
        np.seterr(divide='ignore', invalid='ignore')
        pca = PCA(n_components=npc)
        pca.fit(vec)
        return pca.components_


def text_class(text):
    """
    获取需要预测的文本分类的已有文本
    :param text: 需要预测的文本
    :return: 文本分类已有文本
    """
    sif_obj = SifEmbedding(text)
    sentence_vec = sif_obj.sentences_vec
    sentence_class = str(kmeans.predict([sentence_vec])[0])
    return sentence_vec, bool_json[sentence_class]


def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))


def get_answer(text):
    """
    获取某个问题的答案
    :param text: 问题
    :return:
    """
    sentence_vec, sentence_class = text_class(text)

    corr_list = list()
    for sentence in sentence_class:
        corr = eucliDist(np.array(sentence_vec), sentence_vecs[sentence])
        if corr <= 0.002:
            corr_list.append(sentence)

    if not corr_list: return False

    corr_dict = dict()
    text = content_process(text)
    for i in corr_list:
        corr_i = 0
        question = questions[i]
        question = content_process(question)
        for j in text:
            corr_i = corr_i + 1 if j in question else corr_i
        corr_dict[answers[i]] = corr_i

    answer = max(corr_dict, key=corr_dict.get)
    if corr_dict[answer] == 0:
        return False
    else:
        return max(corr_dict, key=corr_dict.get)
