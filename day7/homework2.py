#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


from io import open
import glob
import os
import random
import unicodedata
import string
import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt


def find_files(path):
    """
    获取目标路径下符合条件的所有文件
    :param path: 目标路径与文件匹配条件
    :return: 一个可迭代对象
    """
    return glob.iglob(path)

# print(find_files('data/names/*.txt'))

# a-z A-Z
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicode_2_Ascii(s):
    """
    将unicode转换成标准的Ascii字符
    :param s: 需要转换的字符串
    :return:
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


category_lines = {}     # 分类对应内容字典
all_categories = []     # 所有的分类


def read_lines(filename):
    """
    读文件转换成Ascii字节列表
    :param filename:
    :return:
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_2_Ascii(line) for line in lines]

for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


def letter_to_index(letter):
    """
    查找某个字符的下标
    :param letter: 要查找的字符
    :return: 下标
    """
    return all_letters.find(letter)


def letter_to_tensor(letter):
    """
    为单个字符做独热
    :param letter: 单字符
    :return:
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    """
    循环line的每个字符做独热
    :param line:
    :return:
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


class RNN(nn.GRU):
    def __init__(self, input_size, hidden_size, output_size):
        """

        :param input_size: 输入类型个数
        :param hidden_size: 隐藏层个数
        :param output_size: 输出类型个数
        """
        super(RNN, self).__init__(input_size, hidden_size)        # LSTM or GRU

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 1024
rnn = RNN(n_letters, n_hidden, n_categories)


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def sample(l):
    return l[random.randint(0, len(l) - 1)]


def sample_trainning():
    category = sample(all_categories)
    line = sample(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


criterion = nn.CrossEntropyLoss()
learning_rate = 0.005


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    return output, loss.item()


n_iters = 1000
print_every = 500
plot_every = 100


# Keep track of losses for plotting
current_loss = 0
all_losses = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = sample_trainning()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.plot(all_losses)
plt.savefig(f'./homework3_{n_hidden}.png')

