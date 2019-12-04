#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


file_path = "../data/export_sql_1558435/"
new_file_name = 'content.csv'
df = pd.read_csv(file_path+new_file_name, header=None, names=['id', 'content'])
with open('../../data/wiki_and_content', 'w+', encoding='utf-8') as f:
    for i in df['content']:
        f.write(i)
        f.write('\n')


wiki_file = '../../data/wiki_and_content'
model = Word2Vec(LineSentence(wiki_file), size=200, sg=1, workers=multiprocessing.cpu_count())
model.save('../../models/wiki/news.model')
model.wv.save_word2vec_format('../../models/wiki/news.model.wv.vectors.npy')
