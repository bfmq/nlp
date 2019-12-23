#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import pandas as pd
import re
import math
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from gensim import corpora, models
import jieba.posseg as jp, jieba


data_source = "../day5/data/export_sql_1558435/sqlResult_1558435.csv"
data = pd.read_csv(data_source, encoding='utf-8')
data = data.fillna('')
content = data['content'].tolist()


def cut(string):
    return ' '.join(jieba.cut(string))


def token(string):
    return re.findall(r'[\d|\w]+', string)

news_content = [token(n) for n in content]
news_content = [''.join(n) for n in news_content]
news_content = [cut(n) for n in news_content]


def document_frequency(word):
    """
    计算某词汇在总文档文章里出现过的次数
    :param word: 某词汇
    :return:
    """
    return sum(1 for n in news_content if word in n)


def idf(word):
    """
    计算某词汇出现过的文章数与总文档的倒数求log
    :param word: 某词汇
    :return:
    """
    return math.log10(len(news_content)/document_frequency(word))


def tf(word, document):
    """
    计算某词汇在该文章中的词频
    :param word: 某词汇
    :param document: 一篇文章
    :return:
    """
    words = document.split()
    return sum(1 for w in words if w == word)


def tf_idf(word, document):
    return tf(word, document) * idf(word)


def get_keywords_of_a_document(document):
    """
    获取某文章所有词汇的tf-idf值并倒序
    :param document:
    :return:
    """
    words = set(document.split())
    tfidf = [(w, tf_idf(w, document)) for w in words]
    tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
    return tfidf


r = get_keywords_of_a_document(news_content[1])
print(r)

vectorizer = TfidfVectorizer(max_features=10000)
sample_num = 50000
sub_samples = news_content[:sample_num]
X = vectorizer.fit_transform(sub_samples)
document_1, document_2 = random.randint(0, 1000), random.randint(0, 1000)
vector_of_document_1, vector_of_document_2 = X[document_1].toarray()[0], X[document_2].toarray()[0]


def distance(v1, v2):
    return cosine(v1, v2)

r = distance(vector_of_document_1,vector_of_document_2)
print(r)

##################################################################################
flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')
news = data["content"][:100]
stop_words = []
with open('../day5/data/stop/stopword.txt', encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.append(word.strip())

words_ls = []
for text in news:
    words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stop_words]
    words_ls.append(words)

dictionary = corpora.Dictionary(words_ls)
corpus = [dictionary.doc2bow(words) for words in words_ls]
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)
for topic in lda.print_topics(num_words=10):
    print(topic)

text = data["content"][102]
words = [[w.word for w in jp.cut(text) if w.flag in flags and w.word not in stop_words]]
text_corpus = [dictionary.doc2bow(word) for word in words]
print(list(lda[text_corpus]))
