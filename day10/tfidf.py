#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import jieba
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


file_path = "./data/"
new_file_name = 'content.csv'
source_pd = pd.read_csv(file_path+new_file_name)[0:20000]
test_pd = pd.read_csv(file_path+new_file_name)[21000:25000]


def get_data(df):
    X = []
    Y = []
    for row in df.itertuples():
        try:
            x = getattr(row, 'content')
            y = getattr(row, 'source')
            word_list = list(jieba.cut(x))
            words = [wl for wl in word_list]
            X.append(' '.join(words))
            Y.append(y)

        except Exception as e:
            continue
    return X, Y


with open('./data/stopword.txt', 'rb') as f:
    STOP_WORDS = [line.strip() for line in f.readlines()]

X, Y = get_data(source_pd)
X_test, Y_test = get_data(test_pd)

# 计算tfidf矩阵
tt = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5)
tfidf = tt.fit_transform(X)

test_tf = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5, vocabulary=tt.vocabulary_)
tfidf_test = test_tf.fit_transform(X_test)


svm = SVC()
ada50 = AdaBoostClassifier()
ada100 = AdaBoostClassifier(n_estimators=100)
knn5 = KNeighborsClassifier(algorithm='ball_tree')
knn9 = KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree', leaf_size=50)
rdf = RandomForestClassifier()

svm.fit(tfidf, Y)
ada50.fit(tfidf, Y)
ada100.fit(tfidf, Y)
knn5.fit(tfidf, Y)
knn9.fit(tfidf, Y)
rdf.fit(tfidf, Y)

# 训练完成后保存模型
joblib.dump(svm, './models/tfidf/svm.pkl')
joblib.dump(ada50, './models/tfidf/ada50.pkl')
joblib.dump(ada100, './models/tfidf/ada100.pkl')
joblib.dump(knn9, './models/tfidf/knn9.pkl')
joblib.dump(knn5, './models/tfidf/knn5.pkl')
joblib.dump(rdf, './models/tfidf/rdf.pkl')

# 加载模型
svm = joblib.load('./models/tfidf/svm.pkl')
ada50 = joblib.load('./models/tfidf/ada50.pkl')
ada100 = joblib.load('./models/tfidf/ada100.pkl')
knn9 = joblib.load('./models/tfidf/knn9.pkl')
knn5 = joblib.load('./models/tfidf/knn5.pkl')
rdf = joblib.load('./models/tfidf/rdf.pkl')

# 获取预测结果
svm_predict = svm.predict(tfidf_test)
ada50_predict = ada50.predict(tfidf_test)
ada100_predict = ada100.predict(tfidf_test)
knn9_predict = knn9.predict(tfidf_test)
knn5_predict = knn5.predict(tfidf_test)
rdf_predict = rdf.predict(tfidf_test)

svm_score = accuracy_score(Y_test, svm_predict) * 100
ada50_score = accuracy_score(Y_test, ada50_predict) * 100
ada100_score = accuracy_score(Y_test, ada100_predict) * 100
knn9_score = accuracy_score(Y_test, knn9_predict) * 100
knn5_score = accuracy_score(Y_test, knn5_predict) * 100
rdf_score = accuracy_score(Y_test, rdf_predict) * 100

# ada50效果是比较好的
bar_x = ['svm', 'ada50', 'ada100', 'knn9', 'knn5', 'rdf']
bar_y = [svm_score, ada50_score, ada100_score, knn9_score, knn5_score, rdf_score]
# bar_y = [100.0, 96.15, 95.875, 80.575, 75.5, 80.2]

plt.bar(bar_x, bar_y, color='g', align='center')
plt.xlabel('model')
plt.ylabel('score%')
plt.savefig('./tfidf_score.png')
