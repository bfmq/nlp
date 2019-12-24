#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

stop_words = []

with open('../day5/data/stop/stopword.txt', encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.append(word.strip())

stop_words = set(stop_words)
print('\n' in stop_words)
# with open('../day5/data/stop/stopword1.txt', 'w',encoding='utf-8') as f:
#     for stop_word in stop_words:
#         f.write(stop_word)
#         f.write('\n')
