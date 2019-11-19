#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def tf_reset():
    tf.reset_default_graph()
    return tf.Session()

# 创建一个sess，类似于创造一个运行环境空间
sess = tf_reset()

# 常量constant
a = tf.constant(1.0)
b = tf.constant(2.0)

# 操作加
c = a + b

# 获取结果需要使用环境运行XXX操作，操作会去找使用到的变量
c_run = sess.run(c)

# print('c = {0}'.format(c_run))
#############################################


# 占位变量placeholder，dtype变量类型，shape维度，name为命名在变量很多时方便查询使用
a = tf.placeholder(dtype=tf.float32, shape=[1], name='a_placeholder')
b = tf.placeholder(dtype=tf.float32, shape=[1], name='b_placeholder')

# 给占位变量赋予不同的feed_dict进行运行，这样需要定义的变量会减少
c0_run = sess.run(c, feed_dict={a: [1.0], b: [2.0]})
c1_run = sess.run(c, feed_dict={a: [2.0], b: [4.0]})

# print('c0 = {0}'.format(c0_run))
# print('c1 = {0}'.format(c1_run))
#############################################


# 占位变量的shape维度也可以不用预先定义
a = tf.placeholder(dtype=tf.float32, shape=[None], name='a_placeholder')
b = tf.placeholder(dtype=tf.float32, shape=[None], name='b_placeholder')

# 这样placeholder变量可以变得更灵活，需要定义的变量又可以减少
c0_run = sess.run(c, feed_dict={a: [1.0], b: [2.0]})
c1_run = sess.run(c, feed_dict={a: [1.0, 2.0], b: [2.0, 4.0]})

# print(a)
# print('a shape: {0}'.format(a.get_shape()))
# print(b)
# print('b shape: {0}'.format(b.get_shape()))
# print('c0 = {0}'.format(c0_run))
# print('c1 = {0}'.format(c1_run))
#############################################


a = tf.constant([[-1.], [-2.], [-3.]], dtype=tf.float32)
b = tf.constant([[1., 2., 3.]], dtype=tf.float32)

# 这个run是干了什么没看明白
# 好像等于就是把ab运行到环境中了
a_run, b_run = sess.run([a, b])

c = b + b

c_run = sess.run(c)

# print('a:\n{0}'.format(a_run))
# print('b:\n{0}'.format(b_run))
# print('c:\n{0}'.format(c_run))
#############################################


# 矩阵乘法
c_elementwise = a * b
# 点乘
c_matmul = tf.matmul(b, a)

c_elementwise_run, c_matmul_run = sess.run([c_elementwise, c_matmul])

# print('a:\n{0}'.format(a_run))
# print('b:\n{0}'.format(b_run))
# print('c_elementwise:\n{0}'.format(c_elementwise_run))
# print('c_matmul: \n{0}'.format(c_matmul_run))
#############################################

# 用之前的结果再运算
c0 = b + b
# c1 = b + b + 1
c1 = c0 + 1

c0_run, c1_run = sess.run([c0, c1])

# print('b:\n{0}'.format(b_run))
# print('c0:\n{0}'.format(c0_run))
# print('c1:\n{0}'.format(c1_run))
#############################################

# 算平均数
c = tf.reduce_mean(b)

c_run = sess.run(c)

# print('b:\n{0}'.format(b_run))
# print('c:\n{0}'.format(c_run))
##########################################################################################


b = tf.constant([[1., 2., 3.]], dtype=tf.float32)
b_run = sess.run(b)
# print('b:\n{0}'.format(b_run))

# 把值赋予变量，类似于创建一个对应关系，但是并没有初始化
var_init_value = [[2.0, 4.0, 6.0]]
var = tf.get_variable(name='myvar',
                      shape=[1, 3],
                      dtype=tf.float32,
                      initializer=tf.constant_initializer(var_init_value))
# print(var)

c = b + var

# print(b)
# print(var)
# print(c)

# 初始化
init_op = tf.global_variables_initializer()
sess.run(init_op)
c_run = sess.run(c)

# print('b:\n{0}'.format(b_run))
# print('var:\n{0}'.format(var_init_value))
# print('c:\n{0}'.format(c_run))
##########################################################################################

# [:, None] = [:, np.newaxis]新增加一个维度
inputs = np.linspace(-2*np.pi, 2*np.pi, 10000)[:, None]
outputs = np.sin(inputs) + 0.05 * np.random.normal(size=[len(inputs),1])
# plt.scatter(inputs[:, 0], outputs[:, 0], s=0.1, color='k', marker='o')
# plt.show()


def create_model():
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    # 初始化权重矩阵
    W0 = tf.get_variable(name='W0', shape=[1, 20], initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.get_variable(name='W1', shape=[20, 20], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name='W2', shape=[20, 1], initializer=tf.contrib.layers.xavier_initializer())

    # 初始化常量矩阵
    b0 = tf.get_variable(name='b0', shape=[20], initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[20], initializer=tf.constant_initializer(0.))
    b2 = tf.get_variable(name='b2', shape=[1], initializer=tf.constant_initializer(0.))

    weights = [W0, W1, W2]
    biases = [b0, b1, b2]
    activations = [tf.nn.relu, tf.nn.relu, None]

    # layer0 = relu(W0  .* input + b0)
    # layer1 = relu(W1  .* layer0 + b1)
    # layer2 = W2  .* layer1 + b2
    # 这个循环的本质就是简化了流程，用循环代替了原本流程
    layer = input_ph
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    output_pred = layer

    return input_ph, output_ph, output_pred


input_ph, output_ph, output_pred = create_model()

# tf.reduce_mean计算平均值
# tf.square平方
# 计算真实值与预测值的损失值
mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

# 优化器
opt = tf.train.AdamOptimizer().minimize(mse)

# 初始化变量
sess.run(tf.global_variables_initializer())
# 保存模型
saver = tf.train.Saver()

# run training
batch_size = 32
for training_step in range(10000):
    # get a random subset of the training data
    indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
    input_batch = inputs[indices]
    output_batch = outputs[indices]

    # run the optimizer and get the mse
    # _, mse_run = sess.run([opt, mse], feed_dict={input_ph: input_batch, output_ph: output_batch})

    # print the mse every so often
    # if training_step % 1000 == 0:
    #     print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
    #     saver.save(sess, 'model.ckpt')

##########################################################################################


# 从之前已经训练好的模型恢复
saver.restore(sess, "model.ckpt")

output_pred_run = sess.run(output_pred, feed_dict={input_ph: inputs})

# plt.scatter(inputs[:, 0], outputs[:, 0], c='k', marker='o', s=0.1)
# plt.scatter(inputs[:, 0], output_pred_run[:, 0], c='r', marker='o', s=0.1)
# plt.show()
##########################################################################################


a = tf.constant(np.random.random((4, 1)))
b = tf.constant(np.random.random((1, 4)))
c = a * b
assert c.get_shape() == (4, 4)

a = tf.get_variable('I_am_a_variable', shape=[4, 6])
b = tf.get_variable('I_am_a_variable_too', shape=[2, 7])
for var in tf.global_variables():
    print(var.name)


# 获取帮助介绍
# print(help(tf.reduce_mean))
# print(help(tf.contrib.layers.fully_connected))
# print(help(tf.contrib.layers.fully_connected))

# 使用gpu
# gpu_device = 0
# gpu_frac = 0.5
#
# # make only one of the GPUs visible
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
#
# # only use part of the GPU memory
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
# config = tf.ConfigProto(gpu_options=gpu_options)
#
# # create the session
# tf_sess = tf.Session(graph=tf.Graph(), config=config)
