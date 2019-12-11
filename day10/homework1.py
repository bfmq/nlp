#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from day10.get_sentences import SifEmbedding


file_path = "./data/"
new_file_name = 'content.csv'
source_pd = pd.read_csv(file_path+new_file_name)[21000:25000]
X = []
Y = []

# 获取原标签
for row in source_pd.itertuples():
    try:
        x = getattr(row, 'content')
        y = getattr(row, 'source')
        sif = SifEmbedding(x, None)
        X.append(sif.sentences_vec)
        Y.append(y)

    except Exception as e:
        continue

# 加载模型
svm = joblib.load('./models/svm.pkl')
ada50 = joblib.load('./models/ada50.pkl')
ada100 = joblib.load('./models/ada100.pkl')
knn9 = joblib.load('./models/knn9.pkl')
knn5 = joblib.load('./models/knn5.pkl')
rdf = joblib.load('./models/rdf.pkl')

# 获取预测结果
svm_predict = svm.predict(X)
ada50_predict = ada50.predict(X)
ada100_predict = ada100.predict(X)
knn9_predict = knn9.predict(X)
knn5_predict = knn5.predict(X)
rdf_predict = rdf.predict(X)

svm_score = accuracy_score(Y, svm_predict) * 100
ada50_score = accuracy_score(Y, ada50_predict) * 100
ada100_score = accuracy_score(Y, ada100_predict) * 100
knn9_score = accuracy_score(Y, knn9_predict) * 100
knn5_score = accuracy_score(Y, knn5_predict) * 100
rdf_score = accuracy_score(Y, rdf_predict) * 100

bar_x = ['svm', 'ada50', 'ada100', 'knn9', 'knn5', 'rdf']
bar_y = [svm_score, ada50_score, ada100_score, knn9_score, knn5_score, rdf_score]
plt.bar(bar_x, bar_y, color='g', align='center')
plt.xlabel('model')
plt.ylabel('score%')
plt.show()
plt.savefig('./model_score.png')
