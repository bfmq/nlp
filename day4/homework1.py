#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-
# 准确率只有50%，不知道是哪里错了
# 还有training_accuracy是指什么？训练集的准确率在哪步计算，也是跟测试集一样预测完对比？

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

# for i in range(1,11):
#     plt.subplot(2,5,i)
#     plt.imshow(digits.data[i-1].reshape([8,8]),cmap=plt.cm.gray_r)
#     plt.text(3,10,str(digits.target[i-1]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
y_train[y_train < 5 ] = 0
y_train[y_train >= 5] = 1
y_test[y_test < 5] = 0
y_test[y_test >= 5] = 1

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# print("sigmoid([0,2]) = " + str(sigmoid(np.array([0,2]))))


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
    A = sigmoid(y)       # 预测值
    # print(A)

    Y = Y.reshape(1, -1)
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))      # 与真实值的均方误差

    dw = 1/m * np.dot((A-Y), X).T
    db = 1/m * np.sum(A-Y)

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
    A = sigmoid(y)

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
    y_prediction = Y_prediction[0]
    y_prediction[y_prediction < 0.5] = 0
    y_prediction[y_prediction >= 0.5] = 1
    score = accuracy_score(Y_test, y_prediction)
    return {"w":len(w), "b":b, "training_accuracy": None, "test_accuracy":score, "cost":costs}


r = model(X_train, y_train, X_test, y_test, 5000, 1e-3, True)
print(r)


def different_learning_rate():
    for i in range(10):
        learning_rate = 10 ** -i
        r = model(X_train, y_train, X_test, y_test, 500, learning_rate, True)
        plt.scatter(i, r['cost'])
    plt.show()


# different_learning_rate()


def different_iteration_num():
    for iteration_num in range(100, 3000, 100):
        model(X_train, y_train, X_test, y_test, iteration_num, 1e-3, True)


# different_iteration_num()
