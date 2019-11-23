#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

import numpy as np


def conv_single_step(a_prev_slice, W, b):
    """
    计算单步卷积操作
    :param a_prev_slice: 原单步的一个小矩阵
    :param W: 卷积核
    :param b: 常量
    :return: 经过卷积核计算后结果
    """
    # 与卷积核点乘后的小矩阵
    s = a_prev_slice * W

    # 小矩阵数值总和
    Z = np.sum(s)

    # 加伤常量
    Z += b

    return Z

# np.random.seed(1)
# a_slice_prev = np.random.randn(4, 4, 3)
# W = np.random.randn(4, 4, 3)
# b = np.random.randn(1, 1, 1)
#
# Z = conv_single_step(a_slice_prev, W, b)
# print("Z =", Z)


def zero_pad(X, pad):
    """
    为角落里的像素提高使用次数，增加边缘信息，填充0可以保持卷积计算前后卷的高和宽不变化
    :param X: 原维度
    :param pad: 填充值
    :return: 填充后的维度
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    return X_pad


# np.random.seed(1)
# x = np.random.randn(4, 3, 3, 2)
# x_pad = zero_pad(x, 2)
# print ("x.shape =\n", x.shape)
# print ("x_pad.shape =\n", x_pad.shape)
# print ("x[1,1] =\n", x[1,1])
# print ("x_pad[1,1] =\n", x_pad[1,1])


def conv_forward(A_prev, W, b, hparameters):
    """
    将整个矩阵分别进行conv_single_step，这样最终获得的就是卷积后的整个矩阵
    :param A_prev: 完全的原矩阵
    :param W: 卷积核
    :param b: 常量
    :param hparameters: 其他参数，如步长、填充空白
    :return: 经过卷积核后的矩阵
    """
    # 原矩阵维度
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 卷积核维度
    (f, f, n_C_prev, n_C) = W.shape

    # 获取hparameters里的步长与填充值
    stride = hparameters.get('stride', 1)
    pad = hparameters.get('pad', 0)

    # 步长与填充值会影响我们的单步卷积的取值，提前计算好
    n_H = int((n_H_prev - f + 2*pad) / stride + 1)
    n_W = int((n_W_prev - f + 2*pad) / stride + 1)

    # 预设应返回的经过卷积核后的矩阵大小
    Z = np.zeros((m, n_H, n_W, n_C))

    # 将Z进行填充操作
    A_prev_pad = zero_pad(A_prev, pad)

    # 4步循环等于把整个元数据的每个维度取了一遍
    for i in range(m):  # 循环所有
        a_prev_pad = A_prev_pad[i, :, :, :]  # Select ith training example's padded activation
        for h in range(n_H):    # 类似于循环高
            vert_start = stride * h
            vert_end = vert_start + f

            for w in range(n_W):  # 类似于循环宽
                horiz_start = stride * w
                horiz_end = horiz_start + f

                for c in range(n_C):  # 循环输出的通道数

                    # 获取单步矩阵
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    weights = W[:,:,:,c]
                    biases = b[:,:,:,c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    assert (Z.shape == (m, n_H, n_W, n_C))

    cache = (A_prev, W, b, hparameters)

    return Z, cache


# np.random.seed(1)
# A_prev = np.random.randn(10,5,7,4)
# print(A_prev.shape)
# W = np.random.randn(3,3,4,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 1,
#                "stride": 2}
#
# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("Z's mean =\n", np.mean(Z))
# print("Z[3,2,1] =\n", Z[3,2,1])
# print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])


def pool_forward(A_prev, hparameters, mode="max"):
    """
    与卷积层类似，只是池化层选取后计算的矩阵的平均值或者最大值
    :param A_prev: 完全的原矩阵
    :param hparameters: 其他参数，如步长、填充空白
    :param mode: 选用池化的类型average/max
    :return:
    """
    # 原矩阵维度
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # 预设应返回的经过卷积核后的矩阵大小
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # 循环所有
        for h in range(n_H):  # 类似于循环高
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):  # 类似于循环宽
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):  # 循环输出的通道数
                    # 获取单步矩阵
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end,c]

                    # 为单步矩阵做对应选择运算
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    # 为池化层的反向传输储存元数据
    cache = (A_prev, hparameters)

    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache

np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 1, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A.shape = " + str(A.shape))
print("A =\n", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A =\n", A)
