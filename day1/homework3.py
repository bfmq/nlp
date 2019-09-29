#!/usr/bin/env python
# -*- coding:utf8 -*-
# __author__ = '北方姆Q'
# __datetime__ = 2019/9/29 17:25


from nlp.day1.homework1 import get_generation_by_gram, activity_2
from nlp.day1.homework2 import two_gram_model


def generate_best(grammar_rule, target):
    # 生成句子列表
    generate_list = get_generation_by_gram(grammar_rule, target)
    # 获取每个句子的分值
    score_list = []
    for generate in generate_list:
        score_list.append(two_gram_model(generate))

    # 返回排序好的[(句子,分)，(句子,分)]
    return sorted(zip(score_list, generate_list), reverse=True)


r = generate_best(activity_2, "activity")
print(r)
print(r[0][1])
