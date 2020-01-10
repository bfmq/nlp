#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

import os
from sklearn.metrics import accuracy_score, f1_score
from keras import optimizers, regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import pandas as pd
import numpy as np
from keras import backend as K

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# K.tensorflow_backend._get_available_gpus()

# 读取所有数据
train = pd.read_csv('../../data/ai_challenger_sentiment_analysis/train_char.csv')
val = pd.read_csv('../../data/ai_challenger_sentiment_analysis/validation_char.csv')
test = pd.read_csv('../../data/ai_challenger_sentiment_analysis/test_char.csv')
data = pd.read_csv('../../data/ai_challenger_sentiment_analysis/data_char.csv')
embedding_matrix = np.load('../../models/ai_challenger_sentiment_analysis/embedding_matrix_char_all.npy')
texts = data['content']
maxlen = 1359
max_words = 50000

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(texts)
vocab = tokenizer.word_index
data_w = tokenizer.texts_to_sequences(texts)
data_T = sequence.pad_sequences(data_w, maxlen=1359)

new_train = data_T[:105000]
new_val = data_T[105000:120000]
new_test = data_T[120000:]


def build_model():
    model = Sequential()
    model.add(Embedding(8291, embedding_matrix.shape[1], weights=[embedding_matrix], input_length=maxlen, trainable=True))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add(Conv1D(128, 4, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add(Conv1D(256, 5, padding='same', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
    return model


def train_model(train_x=new_train, test_x=new_test, val_x=new_val, y_col='location_traffic_convenience', folds=2):
    result = {}
    model = build_model()

    # 编译优化器
    adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    train_y_onehot = pd.get_dummies(train[y_col])[[-2, -1, 0, 1]].values
    y_val_onehot = pd.get_dummies(val[y_col])[[-2, -1, 0, 1]].values
    y_val = val[y_col] + 2
    y_test_pred = 0

    # 提前停止条件
    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, mode='auto')

    # 学习率优化
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

    # 保存更好的模型
    best_weights_filepath = 'models/cnn_2/best_weights.hdf5'
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', save_best_only=True, mode='auto')

    for i in range(folds):
        model.fit(train_x, train_y_onehot, epochs=5, batch_size=64, validation_data=(val_x, y_val_onehot),
                            callbacks=[earlyStopping, saveBestModel, reduce_lr], shuffle=True, use_multiprocessing=True,
                            workers=4)

        # reload best weights
        model.load_weights(best_weights_filepath)

        # 预测验证集和测试集
        y_val_pred = model.predict(val_x)
        y_test_pred += model.predict(test_x)
        y_val_pred = np.argmax(y_val_pred, axis=1)

        F1_score = f1_score(y_val_pred, y_val, average='macro')
        print(y_col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, y_val))

    y_test_pred = np.argmax(y_test_pred, axis=1)
    result[y_col] = y_test_pred - 2
    return result, model

if "__main__" == __name__:
    y_cols = train.columns[2:]
    for y_col in y_cols:        # 循环训练出20个对应四分类模型保存
        result, model = train_model(y_col=y_col, folds=2)
        model.save(f'../../models/ai_challenger_sentiment_analysis/{y_col}.h5')
