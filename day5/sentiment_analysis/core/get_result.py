#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# 加载停用词表
stop_words = ['\n', '\r\n', '\r']
with open('data/stop/stopword.txt', encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.append(word.strip())


def content_process(content):
    return [c for c in content if c not in stop_words]


test = pd.read_csv('data/ai_challenger_sentiment_analysis/test/testa.csv')
data = pd.read_csv('data/ai_challenger_sentiment_analysis/data_char.csv')
texts = data['content']
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(texts)
data_w = tokenizer.texts_to_sequences(texts)


def get_result(content):
    content = content_process(content)
    content = [f"'{c}'" for c in content]
    w = tokenizer.texts_to_sequences([content])
    t = sequence.pad_sequences(w, maxlen=1359)

    result = {}
    y_cols = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find', 'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 'price_level', 'price_cost_effective', 'price_discount', 'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness', 'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation', 'others_overall_experience', 'others_willing_to_consume_again']

    for y_col in y_cols:
        model = load_model(f'models/ai_challenger_sentiment_analysis/{y_col}.h5')
        y = model.predict(t)
        y = np.argmax(y, axis=1)[0]
        result[y_col] = int(y-2)
        # print(y)
    return result


# random_df = test.ix[0]
# content = random_df['content']
# pred = json.loads(random_df['pred'])
# print(content)
# print(pred)
# r = get_result(content)
# print(r)
