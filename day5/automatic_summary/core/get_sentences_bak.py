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
from sklearn.decomposition import TruncatedSVD
from math import isnan
from itertools import product


# 模型词频比例计算后保存的json文件，直接读取结果略过运算，节约时间
# word_frequency = json.load(open('automatic_summary/core/word_frequency.json', encoding='utf-8'))
# model = word2vec.Word2Vec.load("models/wiki/ch_corpus.model")

model = word2vec.Word2Vec.load("../../models/wiki/news.model")
length = len(model.wv.vocab.keys())
word_frequency = {w:model.wv.vocab[w].count/length for w in model.wv.vocab}
json.dump(word_frequency, open('./word_frequency.json', 'w',encoding='utf-8'))
max_fre = max(word_frequency.values())
sen_vec = np.zeros_like(model.wv['测试'])
print(max_fre)
print(sen_vec.shape)


def cut(text):
    return ' '.join(jieba.cut(text))


def SIF_sentence_embedding(text, alpha=1e-4):
    sub_sentences, _ = split_sentences(text)
    sub_sentences_vec = np.array([get_weight_average(x, alpha) for x in sub_sentences])

    # SIF公式第二步，去除PCA主成影响
    return remove_pca(sub_sentences_vec)


def get_weight_average(text, alpha=1e-4):
    """

    :param text:
    :param alpha:
    :return:
    """
    # 这里模型是不变的，我就直接用静态数据了
    max_fre = 13.173838755882954
    sen_vec = np.zeros((200,))
    # max_fre = max(word_frequency.values())
    # sen_vec = np.zeros_like(model.wv['测试'])

    words = cut(text).split()
    words = [w for w in words if w in model]

    # SIF公式第一步，（学习率/比重+学习率）* 词向量的加和
    for w in words:
        fre = word_frequency.get(w, max_fre)
        weight = alpha / (fre + alpha)
        sen_vec += weight * model.wv[w]

    if len(words) != 0:     # 吕汶颖优化
        sen_vec /= len(words)
    return sen_vec


def compute_pca(sentences_vec, npc=1):
    """
    使用PCA计算第一主成因素，由吕汶颖提供
    :param sentences_vec:
    :param npc:
    :return:
    """
    np.seterr(divide='ignore', invalid='ignore')
    pca = PCA(n_components=npc)
    pca.fit(sentences_vec)
    return pca.components_


def compute_svd(s_vec,npc=1):
    """
    使用SVD计算第一主成因素，由吕汶颖提供
    :param s_vec:
    :param npc:
    :return:
    """
    np.seterr(divide='ignore', invalid='ignore')
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(s_vec)
    return svd.components_


def remove_pca(sentences_vec, npc=1):
    """
    移除主成因素，由吕汶颖提供
    :param sentences_vec:
    :param npc:
    :return:
    """
    u = compute_pca(sentences_vec, npc)
    if npc == 1:
        sentences_vec -= sentences_vec.dot(u.T) * u
    else:
        sentences_vec -= sentences_vec.dot(u.T).dot(u)
    return sentences_vec


def split_sentences(text, p='[。，,：!！？?;；]', filter_p='\s+'):
    """
    句子分割
    :param text:
    :param p:
    :param filter_p:
    :return:
    """
    punctuate = re.findall(p, text)
    f_p = re.compile(filter_p)
    text = re.sub(f_p, '', text)
    pattern = re.compile(p)
    split = re.split(pattern,text)
    return split[:-1], punctuate


def get_keywords(text, title=None, alpha=1e-4):
    """
    获取标题或者正文本的关键字的分数，由吕汶颖提供
    :param text: 正文本
    :param title: 标题
    :param alpha:
    :return: 关键字的分数
    """
    # 标题效果好，优先使用标题
    text = title if title else text

    text_vec = get_weight_average(text, alpha=alpha).reshape(1, -1)
    text_vec = remove_pca(text_vec)

    words = cut(text).split()
    words = [w for w in words if w in model]

    corr_words_dict = {w: cosine(model.wv[w], text_vec) for w in words}

    length = len(words)
    length_n = int(length * 0.01)
    keywords_score_list = sorted(corr_words_dict.items(), key=lambda x: x[1], reverse=True)[:length_n]

    return [keyword for keyword, _ in keywords_score_list]


def get_corr(text, title, sub_sentences_vec):
    """
    获取句子排分
    :param text:
    :param title:
    :return:
    """
    sub_sentences, _ = split_sentences(text)
    sentences_vec = get_weight_average(text)
    title_vec = get_weight_average(title)
    corr_score_list = []

    for i in range(sub_sentences_vec.shape[0]):
        # 分别获取于正文本、标题之间的得分，按权重得到最后分数
        cosine_score_by_text = cosine(sentences_vec, sub_sentences_vec[i, :])
        cosine_score_by_title = cosine(title_vec, sub_sentences_vec[i, :])
        cosine_score = 0.9 * cosine_score_by_text + 0.1 * cosine_score_by_title
        if not isnan(cosine_score):  # 这里必须用isnan才能判断nan
            corr_score_list.append([cosine_score, sub_sentences[i]])

    corr_score_list = do_keyword(corr_score_list, text, title, alpha=1e-4)
    corr_score_list = do_knn(corr_score_list)
    return sorted(corr_score_list, reverse=True)


def do_keyword(corr_score_list, text, title, alpha):
    """
    为带有关键字的句子加分，由吕汶颖提供
    :param corr_score_list:
    :param text:
    :param title:
    :param alpha:
    :return:
    """
    keywords_list = get_keywords(text, title, alpha)
    if not keywords_list: return corr_score_list        # 如果关键字为空直接返回不运算

    for (i, (score, sub_sentence)), keyword in product(enumerate(corr_score_list), keywords_list):
        if keyword in sub_sentence:
            corr_score_list[i][0] += 0.1

    return corr_score_list


def do_knn(corr_score_list):
    """
    类似于knn思想，平均不同位置的句子比分
    这里需要用到深拷贝
    :param corr_score_list: 原句子分列表
    :return: 平均后的句子分列表
    """
    new_corr_score_list = copy.deepcopy(corr_score_list)
    length = len(corr_score_list)

    if length <= 5: return corr_score_list

    for i in range(length):
        if i == 0:
            new_corr_score_list[i][0] = np.mean([corr_score_list[i+x][0] for x in range(3)])
        elif i == 1:
            new_corr_score_list[i][0] = np.mean([corr_score_list[i+x][0] for x in range(-1, 2)])
        elif i == length-1:
            new_corr_score_list[i][0] = np.mean([corr_score_list[i+x][0] for x in range(-2, 1)])
        elif i == length-2:
            new_corr_score_list[i][0] = np.mean([corr_score_list[i+x][0] for x in range(-1, 2)])
        else:
            new_corr_score_list[i][0] = np.mean([corr_score_list[i+x][0] for x in range(-1, 2)])

    return new_corr_score_list


def get_summarization(text, title):
    """
    总流程
    :param text:
    :param title:
    :return:
    """
    text_length = len(text)
    if text_length <= 20: return text

    max_count = min(text_length//4, 600)          # 超大文本摘要上限就是600
    alpha = 1e-4 if text_length <= 800 else 2e-5    # 吕汶颖优化

    sub_sentences, sub_punctuates = split_sentences(text)
    sub_sentences_vec = SIF_sentence_embedding(text, alpha)
    ranking_sentences = get_corr(text, title, sub_sentences_vec)

    selected_sen = set()
    current_sen = 0

    for _, ranking_sentence in ranking_sentences:
        if current_sen < max_count:
            current_sen += len(ranking_sentence)
            selected_sen.add(ranking_sentence)
        else:
            break

    summarized = [f'{title}。'] if ('原标题' and '核心提示') not in text else []

    for i in range(len(sub_sentences)):
        if sub_sentences[i] in selected_sen:
            summarized.append(f'{sub_sentences[i]}{sub_punctuates[i]}')

    # 处理最后的标点
    if not summarized[-1].endswith(('。', '.', '！', '!', '？', '?')):
        summarized[-1] = summarized[-1][:-1] + '。'
    return ''.join(summarized)


# file_path = "../../data/export_sql_1558435/"
# file_path = "data/export_sql_1558435/"
# new_file_name = 'content.csv'
# df = pd.read_csv(file_path+new_file_name, header=None, names=['id', 'title', 'content'], nrows=500)

# i = 200
# text = df['content'][i]
# title = df['title'][i]
# r = get_summarization(text, title)
# print(r)
