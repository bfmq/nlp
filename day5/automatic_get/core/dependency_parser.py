#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


from pyltp import Postagger, Parser, Segmentor, SentenceSplitter
import pandas as pd
import random

# data_source = "../../data/export_sql_1558435/content.csv"
# data = pd.read_csv(data_source, encoding='utf-8')
# news = data["content"][463]
news = "马斯克预计，现在如果让12个人登陆火星开始殖民，每人的花费大约是100亿美元，他希望这一数字能够有所降低。" \
       "他说：“如果我们能够将到达火星的费用大致等同于美国的中位数房价（约为20万美元），那么我认为，在火星建立一个自我维持文明的概率非常高。" \
       "马斯克同时认为，不排除成本进一步下降的可能性，比如降低至低于10万美元每人，这一价格比从美国私立大学获得学位便宜得多。" \
       "马斯克降低成本的思路包括，建立在火星上的殖民地能够生产出太空飞船的推进剂，这一推进剂可以让太空飞船在地球与火星之间航行至少一次。" \
       "同时，太空飞船也要有能力在航线上补充燃料。所有这一切多久才会成真？" \
       "“如果一切顺利，那可能在10年以内，但是我不想说这一切，具体会在什么时候发生。”马斯克表示。"
print(news)


class Dependency(object):
    cws_model = "../../models/ltp_data_v3.4.0/cws.model"
    pos_model = "../../models/ltp_data_v3.4.0/pos.model"
    par_model = "../../models/ltp_data_v3.4.0/parser.model"
    ner_model = "../../models/ltp_data_v3.4.0/ner.model"
    similarity = set(['认为', '说道', '回答', '问', '觉得', '知道', '轮不到', '没听说过', '嘲讽地', '却说', '表示', '问道'])

    def __init__(self, sentence):
        self.sentence = sentence
        self.word_list = self.get_word_list()
        self.postag_list = self.get_postag_list()
        self.parser_list = self.get_parser_list()
        self.sentence_list = self.get_sentence_list()

    def get_sentence_list(self):
        """
        获取分句
        :return:
        """
        return SentenceSplitter.split(self.sentence)

    def get_word_list(self):
        """
        获取分词
        :return:
        """
        segmentor = Segmentor()
        segmentor.load(Dependency.cws_model)
        try:
            word_list = list(segmentor.segment(self.sentence))
        except Exception as e:
            word_list = list()
        finally:
            segmentor.release()
            return word_list

    def get_postag_list(self):
        """
        获取词性标注
        :param word_list:
        :return:
        """
        postag = Postagger()
        postag.load(Dependency.pos_model)
        try:
            postag_list = list(postag.postag(self.word_list))
        except Exception as e:
            postag_list = list()
        finally:
            postag.release()
            return postag_list

    def get_parser_list(self):
        """
        获取依存关系
        :param word_list:
        :param postag_list:
        :param model:
        :return:
        """
        parser = Parser()
        parser.load(Dependency.par_model)
        try:
            parsers = parser.parse(self.word_list, self.postag_list)
            parser_list = [(parser.head,parser.relation) for parser in parsers]
        except Exception as e:
            parser_list = list()
        finally:
            parser.release()
            return parser_list

    def get_next(self, index):
        """
        获取下一个词的索引及词
        :param index: 索引
        :return:
        """
        next_word_index, _ = self.parser_list[index]
        return next_word_index, self.word_list[next_word_index]

    def check_similarity(self):
        """
        检测原句中是否包含similarity
        :return: 包含similarity的索引
        """
        similarity_set = set(self.word_list) & Dependency.similarity
        return [idx for idx, word in enumerate(self.word_list) if word in similarity_set] if similarity_set else []


dependency_obj = Dependency(news)
# similarity_index_list = dependency_obj.check_similarity()
similarity_index_list = [61, 76, 198]
for similarity_index in similarity_index_list:
    next_word_index, next_word = dependency_obj.get_next(similarity_index)
    print(next_word_index, next_word)
    print('=============')
