#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

from pyltp import Postagger,Parser,Segmentor
import pandas as pd
import random

data_source = "../day5/data/export_sql_1558435/sqlResult_1558435.csv"
data = pd.read_csv(data_source, encoding='utf-8')
data = data.fillna('')
news = data["content"][463]
news = "她说“今天天气不错，可以出去玩”"

cws_model = "ltp_data_v3.4.0/cws.model"
pos_model = "ltp_data_v3.4.0/pos.model"
par_model = "ltp_data_v3.4.0/parser.model"
ner_model = "ltp_data_v3.4.0/ner.model"


def get_word_list(sentence,model):
    #得到分词
    segmentor = Segmentor()
    segmentor.load(model)
    word_list = list(segmentor.segment(sentence))
    segmentor.release()
    return word_list


def get_postag_list(word_list,model):
    #得到词性标注
    postag = Postagger()
    postag.load(model)
    postag_list = list(postag.postag(word_list))
    postag.release()
    return postag_list


def get_parser_list(word_list,postag_list,model):
    #得到依存关系
    parser = Parser()
    parser.load(model)
    arcs = parser.parse(word_list,postag_list)
    arc_list = [(arc.head,arc.relation) for arc in arcs]
    parser.release()
    return arc_list


ALL = []
def parser(word_index, word_list):
    ALL.append(word_index)
    next_word_index, next_word_link = parser_list[word_index]
    if next_word_index in ALL: return ALL
    return parser(next_word_index, word_list)


word_list = get_word_list(news,cws_model)
postag_list = get_postag_list(word_list,pos_model)
parser_list = get_parser_list(word_list,postag_list,par_model)

for i in range(len(word_list)):
    print(i, word_list[i], parser_list[i])

# next_word_index, next_word_link = parser_list[1]
# print(word_list[1], next_word_index, next_word_link)
# next_word_index, next_word_link = parser_list[318]
# print(word_list[318], next_word_index, next_word_link)
# print(news)
# for i in range(len(word_list)):
#     if word_list[i] == '说':
#         print(i)
        # r = parser(429, word_list)
        # print(r)
