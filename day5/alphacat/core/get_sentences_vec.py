#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import json
import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models import word2vec


word_frequency = json.load(open('../../models/qa/word_frequency.json', encoding='utf-8'))
model = word2vec.Word2Vec.load("../../models/qa/qa50.model")


def cut(text):
    return ' '.join(jieba.cut(text))


class SifEmbedding(object):
    def __init__(self, text):
        """

        :param text: 正文本
        :param title: 标题
        :param alpha: 学习率
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
        # 这里模型是不变的，我就直接用静态数据了
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
        """
        移除主成因素，由吕汶颖提供
        :param vec:
        :param npc:
        :return:
        """
        u = self.compute_pca(vec, npc)
        if npc == 1:
            vec -= vec.dot(u.T) * u
        else:
            vec -= vec.dot(u.T).dot(u)
        return vec

    def compute_pca(self, vec, npc=1):
        """
        使用PCA计算第一主成因素，由吕汶颖提供
        :param vec:
        :param npc:
        :return:
        """
        np.seterr(divide='ignore', invalid='ignore')
        pca = PCA(n_components=npc)
        pca.fit(vec)
        return pca.components_


df = pd.read_csv('../../data/qa/qa_corpus.csv', encoding='utf-8')
questions = df['question']
sentence_vecs = []
for question in questions:
    try:
        sif_obj = SifEmbedding(question)
        sentence_vecs.append(sif_obj.sentences_vec)
    except Exception as e:
        sentence_vecs.append(np.zeros_like(model.wv['银行']))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

json.dump(sentence_vecs, open('./sentence_vecs50.json', 'w', encoding='utf-8'), cls=NpEncoder)

