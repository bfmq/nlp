#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import pyltp
from gensim.models import Word2Vec
from pyltp import Segmentor
import jieba
from gensim.models.word2vec import LineSentence
from pyltp import SentenceSplitter,NamedEntityRecognizer,Postagger,Parser,Segmentor
from gensim import models
import numpy as np
import pandas as pd

data_source = "../day5/data/export_sql_1558435/sqlResult_1558435.csv"
data = pd.read_csv(data_source, encoding='utf-8')
data = data.fillna('')
news = data["content"][0]
print(news)

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


word_list = get_word_list(news,cws_model)
postag_list = get_postag_list(word_list,pos_model)
parser_list = get_parser_list(word_list,postag_list,par_model)
for i in range(len(word_list)):
    print(word_list[i],parser_list[i])



