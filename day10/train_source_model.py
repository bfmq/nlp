#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import json
import pandas as pd
from day10.get_sentences import SifEmbedding
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


# 加载句模型与SVM模型
file_path = "./data/"
new_file_name = 'content.csv'
source_pd = pd.read_csv(file_path+new_file_name)[0:20000]
X = []
Y = []

svm = SVC()
ada50 = AdaBoostClassifier()
ada100 = AdaBoostClassifier(n_estimators=100)
knn5 = KNeighborsClassifier(algorithm='ball_tree')
knn9 = KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree', leaf_size=50)
rdf = RandomForestClassifier()

# 获取句向量作为X，分类为Y训练
for row in source_pd.itertuples():
    try:
        x = getattr(row, 'content')
        y = getattr(row, 'source')
        sif = SifEmbedding(x, None)
        X.append(sif.sentences_vec)
        Y.append(y)

    except Exception as e:
        print(e)
        continue

svm.fit(X, Y)           # svm训练后预测永远为1，不知道是哪里有问题还是维度太多不合适
ada50.fit(X, Y)
ada100.fit(X, Y)
knn5.fit(X, Y)
knn9.fit(X, Y)
rdf.fit(X, Y)

# 训练完成后保存模型
joblib.dump(svm, './models/svm.pkl')
joblib.dump(ada50, './models/ada50.pkl')
joblib.dump(ada100, './models/ada100.pkl')
joblib.dump(knn9, './models/knn9.pkl')
joblib.dump(knn5, './models/knn5.pkl')
joblib.dump(rdf, './models/rdf.pkl')
