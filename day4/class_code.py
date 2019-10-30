#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import resample
from keras.layers import Dense
from keras.models import Sequential


class Node:
    """
    在整个图里的某一节点
    """
    def __init__(self, inputs=list()):
        self.inputs = inputs
        self.outputs = []

        for n in self.inputs:
            n.outputs.append(self)

        self.value = None
        self.gradients = {}

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Input(Node):
    """
    输入节点类
    """
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self] = grad_cost * 1


            # input N --> N1, N2
            # \partial L / \partial N
            # ==> \partial L / \partial N1 * \ partial N1 / \partial N


class Add(Node):
    """
    加运算
    """
    def __init__(self, *nodes):
        Node.__init__(self, nodes)

    def forward(self):
        """
        计算所有node的value总和
        :return:
        """
        self.value = sum(map(lambda n: n.value, self.inputs))


class Linear(Node):
    """
    线性计算节点
    """
    def __init__(self, nodes, weights, bias):
        Node.__init__(self, [nodes, weights, bias])

    def forward(self):
        """
        前向传输
        :return:
        """
        inputs = self.inputs[0].value
        weights = self.inputs[1].value
        bias = self.inputs[2].value

        # self.value = weights * nodes.value + bias
        # y = wx + b
        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            # Get the partial of the cost w.r.t this node.
            grad_cost = n.gradients[self]

            self.gradients[self.inputs[0]] = np.dot(grad_cost, self.inputs[1].value.T)
            self.gradients[self.inputs[1]] = np.dot(self.inputs[0].value.T, grad_cost)
            self.gradients[self.inputs[2]] = np.sum(grad_cost, axis=0, keepdims=False)

            # WX + B / W ==> X
            # WX + B / X ==> W


class Sigmoid(Node):
    """
    将回归问题转换成分类问题
    一个在（0，1）之间连续的对称函数y = 1 / (1 + e^-x)
    """
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-1 * x))

    def forward(self):
        """
        将值转换到（0，1）之间
        :return:
        """
        self.x = self.inputs[0].value
        self.value = self._sigmoid(self.x)

    def backward(self):
        self.partial = self._sigmoid(self.x) * (1 - self._sigmoid(self.x))

        # y = 1 / (1 + e^-x)
        # y' = 1 / (1 + e^-x) (1 - 1 / (1 + e^-x))

        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] = grad_cost * self.partial


class MSE(Node):
    """
    与训练标签比较计算梯度
    """
    def __init__(self, y, a):
        """
        :param y: 训练标签
        :param a: 预测标签
        """
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inputs[0].value.reshape(-1, 1)
        a = self.inputs[1].value.reshape(-1, 1)
        assert (y.shape == a.shape)

        self.m = self.inputs[0].value.shape[0]
        self.diff = y - a

        # 就是上次的sum((y_i - y_hat_i)**2 for y_i, y_hat_i in zip(list(y), list(y_hat)))/len(list(y))
        self.value = np.mean(self.diff ** 2)

    def backward(self):
        self.gradients[self.inputs[0]] = (2 / self.m) * self.diff
        self.gradients[self.inputs[1]] = (-2 / self.m) * self.diff


def forward_and_backward(outputnode, graph):
    """
    正向传输后反向传输
    :param outputnode:
    :param graph: 连接图
    :return:
    """
    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()


def topological_sort(feed_dict):
    """
    这个函数应该可以优化到O(n)，暂时先完成作业
    构造出一个从左到右表示连接关系的列表
    :param feed_dict: 种子字典
    :return: [
                w1
                b1
                w2
                b2
                X
                Y
                w1X + b1
                s1
                w2(s1(w1X + b1)) + b2
                MSE
                ]
    """
    nodes = [n for n in feed_dict]
    S = set(nodes)
    G = dict()
    L = []

    def outputs_append(n, g):
        """
        递归出n节点下所有outputs
        :param n: 某input节点
        :param g: 总图
        :return:
        """
        for m in n.outputs:             # 循环n的输出点
            g[m] = {'in': set(), 'out': set()} if m not in g else g[m]  # m不在图里也直接添加
            g[n]['out'].add(m)
            g[m]['in'].add(n)
            outputs_append(m, g)
        return g

    for n in nodes:               # 循环所有节点
        G[n] = {'in': set(), 'out': set()} if n not in G else G[n]      # n不在图里就添加
        G = outputs_append(n, G)

    while S:
        n = S.pop()
        n.value = feed_dict[n] if isinstance(n, Input) else n.value     # input节点的value就等于它自己
        L.append(n)
        for m in n.outputs:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if not G[m]['in']:
                S.add(m)

    return L


def sgd_update(trainables, learning_rate=1e-2):
    """
    按学习率跟梯度更新参数
    :param trainables: 参数列表
    :param learning_rate: 学习率
    :return:
    """
    for t in trainables:
        t.value += -1 * learning_rate * t.gradients[t]


data = load_boston()
losses = []
X_ = data['data']       # 特征向量
Y_ = data['target']     # 预测结果

# 标准化特征向量 ：(x-x平均)/ x标准差
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)


# X, Y，W1, b1，W2, b2都只是普通的输入值节点
X, Y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

# l1,l2是线性计算节点，s1是Sigmoid转换节点
l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(Y, l2)

feed_dict = {
    X: X_,
    Y: Y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 5000
m = X_.shape[0]                      # 训练集总数据量
print(f"Total number of examples = {m}")
batch_size = 16                      # 每次训练使用数据个数
steps_per_epoch = m // batch_size    # 批次

graph = topological_sort(feed_dict)     # 获取整个传输图
trainables = [W1, b1, W2, b2]           # 这几个就是一直被优化的参数

for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # 将数据集随机切分出我们设置的大小
        X_batch, Y_batch = X.value, Y.value = resample(X_, Y_, n_samples=batch_size)
        # 将图进行正向/反向传输
        _ = None
        forward_and_backward(_, graph)
        # 学习率
        rate = 1e-2
        # 对被优化的参数一直按学习率进行梯度下降
        sgd_update(trainables, rate)
        # 每次都更新总loss值方便打印
        loss += graph[-1].value

    if i % 100 == 0:
        print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))
        losses.append(loss)

#################################################################################

model = Sequential()

model.add(Dense(units=64, activation='sigmoid', input_dim=13))  # 64个神经元的全连接层，激活函数用sigmoid，数据输入形状为13
model.add(Dense(units=30, activation='sigmoid', input_dim=64))
model.add(Dense(units=1))

model.compile(loss='mse',           # 计算损失用均方误差
              optimizer='sgd',      # 优化器用随机梯度下降
              metrics=['mse'])      # 评测函数用均方误差

model.fit(X_, Y_, epochs=5000, batch_size=32)






