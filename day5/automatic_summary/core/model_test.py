#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


from gensim.models import word2vec


model = word2vec.Word2Vec.load("../models/wiki/wiki_corpus.model")
r = model.wv.most_similar('美女')
for i in r:
    print(i)

two_corpus = ["腾讯", "百度"]
r = model.similarity(two_corpus[0], two_corpus[1])
print(r)


def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result

r = analogy('中国', '汉语', '美国')
print(r)
r = analogy('美国', '奥巴马', '美国')
print(r)
