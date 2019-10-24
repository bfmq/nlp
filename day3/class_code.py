#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from collections import Counter, defaultdict


def assmuing_function(x):
    """
    某潜在关系函数
    :param x:
    :return:
    """
    return 13.4 * x + 5 + random.randint(-5, 5)


random_data = np.random.random((20, 2))
X = random_data[:, 0]
# Y = random_data[:, 1]
y = [assmuing_function(x) for x in X]
y = np.array(y)

# plt.scatter(X, Y)   # 原图像
# plt.show()
#
# plt.scatter(X, y)   # 预测图像
# plt.show()


def linear_model():
    def f(x):
        return lr.coef_ * x + lr.intercept_

    lr = LinearRegression()             # 创建一个线性回归模型
    lr = lr.fit(X.reshape(-1, 1), y)    # 用X(函数变量)、y(函数结果)训练模型

    x_test = [[x*0.1] for x in range(10)]
    y_predict = lr.predict(x_test)
    print(y_predict)

    plt.scatter(X, y)                   # 画变量与结果的散点图
    plt.scatter(x_test, y_predict, color='yellow')
    plt.plot(X, f(X), color='red')      # 画由模型预测出的线性图
    plt.show()

# linear_model()

###########################################################

y = np.random.randint(0, 5, 20)


def predict(x, k=5):
    """
    根据未知数据的特征用模型（看余弦值相似度）进行预测，返回最终分类结果
    :param x: 需要预测的数据的特征
    :param k: 选取最近的几个邻居最为分类判断
    :return:  分类结果，0或1或2
    """
    def model(X, y):
        """
        模型本身原始特征及对应分类
        :param X: 特征值
        :param y: 分类结果，0或1或2
        :return: [(X1,y1),(X2,y2),(X3,y3)...]
        """
        return [(Xi, yi) for Xi, yi in zip(X, y)]

    def distance(x1, x2):
        """
        计算x1与x2点的余弦值
        :param x1:
        :param x2:
        :return:
        """
        return cosine(x1, x2)

    most_similars = sorted(model(X, y), key=lambda xi: distance(xi[0], x))[:k]
    print(most_similars)
    return max(Counter(x[1] for x in most_similars))


def knn_test():
    x_test = 567
    y_predict = predict(x_test)
    print(y_predict)

# knn_test()


def knn_model():
    knnc = KNeighborsClassifier()       # 创建一个KNN分类器
    knnc.fit(X.reshape(-1, 1), y)       # 训练模型
    x_test = [[x * 0.1] for x in range(10)]
    y_predict = knnc.predict(x_test)    # 预测
    print(y_predict)

# knn_model()

#################################################


def entropy(elements):
    """
    计算熵
    (各个元素出现次数/总数 * （各个元素出现次数/总数）的对数)的总数取负
    :param elements:
    :return:
    """
    counter = Counter(elements)
    probs = [counter[c] / len(elements) for c in elements]
    return - sum(p * np.log(p) for p in probs)

# print(entropy([1, 1, 1, 1]))
# print(entropy([1, 1, 1, 0]))
# print(entropy([1, 1, 2, 1]))
# print(entropy([1, 2, 3, 4]))

mock_data = {
    'gender': ['F', 'F', 'F', 'F', 'M', 'M', 'M'],
    'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],
    'family_number': [1, 1, 2, 1, 1, 1, 2],
    'bought': [1, 1, 1, 0, 0, 0, 1],
}

dataset = pd.DataFrame.from_dict(mock_data)


def find_test():
    sub_split_1 = dataset[dataset['family_number'] == 1]['bought'].tolist()
    sub_split_2 = dataset[dataset['family_number'] != 1]['bought'].tolist()
    print(sub_split_1)
    print(sub_split_2)
    print(entropy(sub_split_1) + entropy(sub_split_2))
    _sub_split_1 = dataset[dataset['gender'] != 'F']['bought'].tolist()
    _sub_split_2 = dataset[dataset['gender'] != 'M']['bought'].tolist()
    print(_sub_split_1)
    print(_sub_split_2)
    print(entropy(_sub_split_1) + entropy(_sub_split_2))

# find_test()


def find_the_min_spilter(training_data: pd.DataFrame, target: str) -> str:
    """
    寻找分割后最小熵的分割特征
    :param training_data: 数据
    :param target: 目标特征
    :return:
    """
    # 将数据去除目标特征
    x_fields = set(training_data.columns.tolist()) - {target}

    spliter = None
    min_entropy = float('inf')          # 预先设置成正无穷

    for f in x_fields:                  # 循环所有特征名
        values = set(training_data[f])  # 取该特征所有数值做集合
        for v in values:                # 循环该特征所有数值集合
            sub_spliter_1 = training_data[training_data[f] == v][target].tolist()       # 计算等于该数值对目标特征分布
            entropy_1 = entropy(sub_spliter_1)                                          # 计算等于该数值时目标特征分布信息熵
            sub_spliter_2 = training_data[training_data[f] != v][target].tolist()       # 计算不等于该数值对目标特征分布
            entropy_2 = entropy(sub_spliter_2)                                          # 计算不等于该数值时目标特征分布信息熵
            entropy_v = entropy_1 + entropy_2                                           # 计算此数值对目标特征的总信息熵

            if entropy_v <= min_entropy:        # 如果信息熵小于当前最小信息熵则更新信息熵与特征
                min_entropy = entropy_v
                spliter = (f, v)

    print('spliter is: {}'.format(spliter))
    print('the min entropy is: {}'.format(min_entropy))
    print('----------------------')

    return spliter


# find_the_min_spilter(dataset, 'bought')
# find_the_min_spilter(dataset[dataset['family_number'] == 1], 'bought')
# sub_df = dataset[dataset['family_number'] == 1]
# find_the_min_spilter(sub_df[sub_df['gender'] != 'M'], 'bought')


def tree_model():
    # 把字符串转成数字
    gender_map = {'F': 0, 'M': 1}
    income_map = {'-10': 0, '+10': 1}
    dataset['gender'] = dataset['gender'].map(gender_map)
    dataset['income'] = dataset['income'].map(income_map)

    features = ['gender', 'income', 'family_number']
    train_features = dataset[features]
    train_labels = dataset['bought']

    # 建模型训练，用基于信息熵ID3算法
    dtc = DecisionTreeClassifier(criterion='entropy')
    dtc.fit(train_features, train_labels)

    x_test = [[a, b, c] for a, b, c in zip(np.random.randint(0, 2, 20),
                                           np.random.randint(0, 2, 20),
                                           np.random.randint(0, 2, 20))]
    y_predict = dtc.predict(x_test)    # 预测
    print(y_predict)

# tree_model()

########################################################

X = [random.randint(0, 100) for _ in range(100)]
Y = [random.randint(0, 100) for _ in range(100)]
training_data = [[x, y] for x, y in zip(X, Y)]
# plt.scatter(X, Y)
# plt.show()


def kmeans_model():
    cluster = KMeans(n_clusters=6, max_iter=500)    # 创建一个分类数为6，迭代次数为500的模型
    cluster.fit(training_data)          # 训练
    print(cluster.cluster_centers_)     # 获取分类中心点
    print(cluster.labels_)              # 获取各点类别

    # 创建6个类别的对应点的列表
    centers = defaultdict(list)
    for label, location in zip(cluster.labels_, training_data):
        centers[label].append(location)

    # 将6种类别的点染上不同颜色
    color = ['red', 'green', 'grey', 'black', 'yellow', 'orange']
    for i, c in enumerate(centers):
        for location in centers[c]:
            plt.scatter(*location, c=color[i])

    # 分类中心点加大显示
    for center in cluster.cluster_centers_:
        plt.scatter(*center, s=100)

    plt.show()


kmeans_model()
