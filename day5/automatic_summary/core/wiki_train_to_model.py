#!/usr/bin/env python
# coding=utf-8

import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# file = wiki + content
file = ''

model = Word2Vec(LineSentence(file), size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())
model.save('../../models/wiki/news.model')
model.wv.save_word2vec_format('../../models/wiki/news.model.wv.vectors.npy')
