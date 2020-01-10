#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


val = pd.read_csv('../../data/ai_challenger_sentiment_analysis/validation_char.csv')
data = pd.read_csv('../../data/ai_challenger_sentiment_analysis/data_char.csv')
texts = data['content']
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(texts)
data_w = tokenizer.texts_to_sequences(texts)
data_T = sequence.pad_sequences(data_w, maxlen=1359)
new_val = data_T[105000:120000]


def test_CV_CNN(val_x=new_val, y_col='location_traffic_convenience'):
    val_y = val[y_col] + 2
    model = load_model(f'../../models/ai_challenger_sentiment_analysis/{y_col}.h5')

    y_val_pred = model.predict(val_x)
    y_val_pred = np.argmax(y_val_pred, axis=1)

    F1_score = f1_score(y_val_pred, val_y, average='macro')
    return F1_score, accuracy_score(y_val_pred, val_y)


if "__main__" == __name__:
    y_cols = val.columns[2:]
    F1_scores = 0

    for y_col in y_cols:
        F1_score, acc = test_CV_CNN(y_col=y_col)
        print(f"{y_col} accuracy_score is {acc}", f"{y_col} f1_score is {F1_score}")
        F1_scores += F1_score

    print('===============')
    print(f"all F1_score is {F1_scores/20}")
