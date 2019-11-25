#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)


def softmax(z):
    z -= np.max(z)
    return np.exp(z)/sum(np.exp(z))


def initialize_parameters(dim):
    """
    初始化参数w,b
    :param dim: 特征维度数量
    :return:
    """

    w = np.random.randn(dim, 1)
    b = np.random.randint(0, 10)

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# w, b = initialize_parameters(X_train.shape[1])


def propagate(w, b, X, Y):
    """
    :param w: weights
    :param b: bias
    :param X: 训练集
    :param Y: 训练标签
    :return:
    """
    m = X.shape[0]
    y = np.dot(X, w) + b
    A = softmax(y)       # 预测值

    Y = Y.reshape(-1, 1)
    cost = -1/m * np.sum(y * np.log(A))      # 与真实值的均方误差

    dw = 1/m * np.dot(X.T, (A - y))
    db = 1/m * np.sum(A - y, axis=0).reshape(1, -1)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {'dw': dw,
             'db': db}
    return grads, cost

# grads, cost = propagate(w, b, X_train, y_train)


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    优化器，在迭代次数中循环获取新的w、b，并按学习率优化后继续输入模型循环继续优化
    :param w: weights
    :param b: bias
    :param X: 训练集
    :param Y: 训练标签
    :param num_iterations: 迭代次数
    :param learning_rate: 学习率
    :param print_cost: 是否调试打印
    :return:
    """
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w + (-1 * dw) * learning_rate
        b = b + (-1 * db) * learning_rate

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    用训练好的模型optimize的params预测测试集的标签
    :param w: weights
    :param b: bias
    :param X: 测试集
    :return: 预测标签
    """
    m = X.shape[0]

    y = np.dot(X, w) + b
    A = softmax(y)

    Y_prediction = A.T
    assert (Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
    """
    模型
    :param X_train: 训练集
    :param Y_train: 训练标签
    :param X_test:  测试集
    :param Y_test:  测试标签
    :param num_iterations:  迭代数
    :param learning_rate:  学习率
    :param print_cost: 是否调试打印
    :return:
    """
    w, b = initialize_parameters(X_train.shape[1])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_prediction = predict(params['w'], params['b'], X_test)
    return {"w":w, "b":b, "training_accuracy": None, "test_accuracy":None, "cost":costs}


r = model(X_train, y_train, X_test, y_test, 500, 1e-3, False)
# print(r)
