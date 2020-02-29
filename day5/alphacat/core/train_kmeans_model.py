#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import json
import pandas as pd


sentence_vecs100 = json.load(open('../../models/qa/sentence_vecs100.json', encoding='utf-8'))
sentence_vecs50 = json.load(open('../../models/qa/sentence_vecs50.json', encoding='utf-8'))

for i in range(5, 11):
    kmeans = KMeans(n_clusters=i, n_init=50, max_iter=1000)
    kmeans.fit(sentence_vecs100)
    predict_y = kmeans.predict(sentence_vecs100)

    df = pd.read_csv('../../data/qa/qa_corpus.csv', encoding='utf-8')
    questions = df['question']
    d = defaultdict(list)

    for index, v in enumerate(predict_y):
        d[str(v)].append(index)

    json.dump(d, open(f'./bool_100_{i}.json', 'w', encoding='utf-8'))
    joblib.dump(kmeans, f'kmeans_100_{i}.pkl')


for i in range(5, 11):
    kmeans = KMeans(n_clusters=i, n_init=50, max_iter=1000)
    kmeans.fit(sentence_vecs50)
    predict_y = kmeans.predict(sentence_vecs50)

    d = defaultdict(list)

    for index, v in enumerate(predict_y):
        d[str(v)].append(index)

    json.dump(d, open(f'./bool_50_{i}.json', 'w', encoding='utf-8'))
    joblib.dump(kmeans, f'kmeans_50_{i}.pkl')
