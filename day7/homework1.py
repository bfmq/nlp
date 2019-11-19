#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 训练集是5W张32 * 32的RGB3通道彩色图，测试集是1W张
# print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)
# print(x_train[0], y_train[0])

# 特征规范化，标签独热处理
x_train = x_train / 255
x_test = x_test / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 模型定义
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')
])


# 用Adam优化器准确率高一些
opt = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=64), steps_per_epoch=1000, epochs=100, validation_data=(x_test,y_test), workers=100, verbose=1)
# model.save('cifar10_model.h5')
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
