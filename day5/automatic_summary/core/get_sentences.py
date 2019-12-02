#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import jieba
import json
import numpy as np
import pandas as pd
import re
import copy
from scipy.spatial.distance import cosine
from gensim.models import word2vec
from sklearn.decomposition import PCA
from math import isnan
from itertools import product


# 模型词频比例计算后保存的json文件，直接读取结果略过运算，节约时间
word_frequency = json.load(open('automatic_summary/core/word_frequency.json', encoding='utf-8'))
model = word2vec.Word2Vec.load("models/wiki/news.model")
# 预读500行内容
file_name = "data/export_sql_1558435/content.csv"
df = pd.read_csv(file_name, header=None, names=['id', 'title', 'content'], nrows=500)


def cut(text):
    return ' '.join(jieba.cut(text))


class SifEmbedding(object):
    def __init__(self, text, title, alpha=None):
        """

        :param text: 正文本
        :param title: 标题
        :param alpha: 学习率
        """
        self.text = text
        self.title = title
        self.text_length = len(self.text)
        # 如果有特指的学习率则用指定的，采样学习率由吕汶颖测试优化完成
        self.alpha = alpha if alpha else 1e-4 if self.text_length < 800 else 2e-5

        # 正文本过小就没必要处理了
        if not self.text_length <= 20:
            self.sub_sentences, self.sub_punctuates = self.split_sentences(self.text)
            self.sub_sentences_vec = self.sif_sentence_embedding()
            self.sentences_vec = self.get_weight_average(self.text)
            self.title_vec = self.get_weight_average(self.title) if self.title else [None]
            self.sentences_corr_list = self.get_sentences_corr()

    def get_summarization(self):
        """
        获取摘要
        :return:  摘要字符串
        """
        # 正文本过小直接返回
        if self.text_length <= 20: return self.text

        max_count = min(self.text_length//4, 600)   # 如果出现超大文本2400+,摘要上限就是600
        current_sen = 0                     # 当前长度

        summarized = np.zeros_like(self.sub_sentences)
        max_index = 0
        for _, sentence in self.sentences_corr_list:
            if current_sen < max_count:
                index = self.sub_sentences.index(sentence)
                max_index = index if index > max_index else max_index
                summarized[index] = self.sub_sentences[index] + self.sub_punctuates[index]
                current_sen += len(sentence)
            else:
                # 如果选用的最后一句的标点不属于结束标点，则取跟着它最近的一句的最后结束语作为结束
                if self.sub_punctuates[max_index] not in ('。', '.', '！', '!', '？', '?'):
                    for new_index in range(max_index+1, len(self.sub_punctuates)):
                        if self.sub_punctuates[new_index] in ('。', '.', '！', '!', '？', '?'):
                            summarized[new_index] = self.sub_sentences[new_index] + self.sub_punctuates[new_index]
                            break

                break

        # 处理标题结尾标点
        if self.title and not self.title.endswith(('。', '.', '！', '!', '？', '?')):
            self.title += '。'
            return f'{self.title}' + ''.join(summarized)
        else:
            return ''.join(summarized)

    def split_sentences(self, text, p='[。，,：!！？?;；]', filter_p='\s+'):
        """
        切分文本
        :param text: 文本
        :param p:
        :param filter_p:
        :return:
        """
        punctuate = re.findall(p, text)
        f_p = re.compile(filter_p)
        text = re.sub(f_p, '', text)
        pattern = re.compile(p)
        split = re.split(pattern, text)
        return split[:-1], punctuate

    def sif_sentence_embedding(self):
        """
        计算句子向量
        :return:
        """
        sub_sentences_vec = np.array([self.get_weight_average(x) for x in self.sub_sentences])
        # SIF公式第二步，去除PCA主成影响
        return self.remove_pca(sub_sentences_vec)

    def get_weight_average(self, text):
        """
        SIF公式第一步
        :param text:
        :return:
        """
        # 这里模型是不变的，我就直接用静态数据了
        # max_fre = max(word_frequency.values())
        # sen_vec = np.zeros_like(model.wv['测试'])
        max_fre = 10.156099364252306
        sen_vec = np.zeros((200,))

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

    def get_sentences_corr(self):
        """
        获取句子的分值
        :return: 句子分值列表，最终的
        """
        corr_score_list = []

        for i in range(self.sub_sentences_vec.shape[0]):
            # 分别获取于正文本、标题之间的得分，按权重得到最后分数
            sub_sentence_vec = self.sub_sentences_vec[i, :]
            cosine_score_by_text = cosine(self.sentences_vec, sub_sentence_vec)
            cosine_score_by_title = cosine(self.title_vec, sub_sentence_vec) if all(self.title_vec) else 0
            cosine_score = 0.9 * cosine_score_by_text + 0.1 * cosine_score_by_title
            if not isnan(cosine_score):  # 这里必须用isnan才能判断nan
                corr_score_list.append([cosine_score, self.sub_sentences[i]])

        corr_score_list = self.do_keyword(corr_score_list)
        corr_score_list = self.do_knn(corr_score_list)
        return sorted(corr_score_list, reverse=True)

    def do_keyword(self, corr_score_list):
        """
        为带有关键字的句子加分，由吕汶颖提供
        :param corr_score_list: 原分值列表
        :return: 新分值列表
        """
        keywords_list = self.get_keywords()
        # 如果关键字为空直接返回不运算
        if not keywords_list: return corr_score_list

        for (i, (_, sub_sentence)), keyword in product(enumerate(corr_score_list), keywords_list):
            if keyword in sub_sentence:
                corr_score_list[i][0] += 0.1

        return corr_score_list

    def get_keywords(self):
        """
        获取关键字，由吕汶颖提供
        :return: 关键字列表
        """
        text_vec = self.remove_pca(self.title_vec.reshape(1, -1)) \
            if all(self.title_vec) else self.remove_pca(self.sentences_vec.reshape(1, -1))

        words = cut(self.title).split() if self.title else self.text
        words = [w for w in words if w in model]

        corr_words_dict = {w: cosine(model.wv[w], text_vec) for w in words}

        length = len(words)
        length_n = int(length * 0.01)
        keywords_score_list = sorted(corr_words_dict.items(), key=lambda x: x[1], reverse=True)[:length_n]

        return [keyword for keyword, _ in keywords_score_list]

    def do_knn(self, corr_score_list):
        """
        为所有句子做knn平滑，这里需要用到深拷贝否则会影响
        :param corr_score_list: 原分值列表
        :return: 新分值列表
        """
        new_corr_score_list = copy.deepcopy(corr_score_list)
        corr_score_list = [score for score, _ in corr_score_list]
        length = len(new_corr_score_list)
        need_num = 5 if self.text_length >= 2400 else 3

        # 太短了就需要再平滑了
        if length <= need_num * 2 + 1: return new_corr_score_list

        for i in range(length):
            weigth_list = [(1-i**2*0.05) for i in range(-(need_num), need_num + 1)]

            start = i - need_num
            end = i + need_num + 1

            if start < 0:
                weigth_list = weigth_list[-(start):]
                for j in range(-start):
                    weigth_list.append(weigth_list[-1] - 0.01)

                end -= start
                start = 0

            if end > length:
                weigth_list = weigth_list[:len(weigth_list) - (end - length)]
                for j in range(end - length):
                    weigth_list.insert(0, weigth_list[0] - 0.01)

                start -= (end - length)
                end = length

            new_corr_score_list[i][0] = np.mean(np.multiply(np.array(corr_score_list[start:end]), np.array(weigth_list)))

        return new_corr_score_list
