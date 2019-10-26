#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import random
from sklearn.datasets import load_boston


data_set = load_boston()
x, y = data_set['data'], data_set['target']
X_rm = x[:, 5]


def price(rm, k, b):
    return k * rm + b


def loss(y, y_hat):
    """
    绝对值平均
    :param y:
    :param y_hat:
    :return:
    """
    return sum(abs((y_i - y_hat_i)) for y_i, y_hat_i in zip(list(y), list(y_hat)))/len(list(y))


def partial_derivative_k(x, y, y_hat):
    """
    使用绝对值时k的导数
    :param x:
    :param y:
    :param y_hat:
    :return:
    """
    n = len(y)
    gradient = 0
    for x_i, y_i, y_hat_i in zip(list(x), list(y), list(y_hat)):
        gradient += (y_i-y_hat_i) * x_i
    return -1 / n * gradient


def partial_derivative_b(y, y_hat):
    """
    使用绝对值时b的导数
    :param y:
    :param y_hat:
    :return:
    """
    n = len(y)
    gradient = 0
    for y_i, y_hat_i in zip(list(y), list(y_hat)):
        gradient += (y_i-y_hat_i)
    return -1 / n * gradient


def boston_loss():
    k = random.random() * 200 - 100
    b = random.random() * 200 - 100

    learning_rate = 1e-3

    iteration_num = 1000000
    losses = []
    for i in range(iteration_num):
        price_use_current_parameters = [price(r, k, b) for r in X_rm]

        current_loss = loss(y, price_use_current_parameters)
        losses.append(current_loss)
        print("Iteration {}, the loss is {}, parameters k is {} and b is {}".format(i, current_loss, k, b))

        k_gradient = partial_derivative_k(X_rm, y, price_use_current_parameters)
        b_gradient = partial_derivative_b(y, price_use_current_parameters)

        k = k + (-1 * k_gradient) * learning_rate
        b = b + (-1 * b_gradient) * learning_rate
    best_k = k
    best_b = b

boston_loss()
