#!/usr/bin/env python
# -*- coding:utf8 -*-
# __author__ = '北方姆Q'
# __datetime__ = 2019/9/29 16:36

import random


activity = """
activity = 人物 时间 动作 地点
人物 = 我 | 他们 | 我们 | 大家
动作 = 去 | 想要 | 想找个
地点 = 商场 | 酒吧 | 家里 | KTV
时间 = 周三 | 下班后 | 七点半| 晚上九点
"""

activity_2 = """
activity = 人物 时间* 动作 地点*
时间* = 时间* 时间 | 时间
地点* = 人物* 地点 | 地点
人物* = 人物* 人物 | 人物
人物 = 我 | 他们 | 我们 | 大家
动作 = 去 | 想要 | 想找个
地点 = 商场 | 酒吧 | 家里 | KTV
时间 = 周三 | 下班后 | 七点半| 晚上九点
"""


simple_grammar = """
sentence => noun_phrase verb_phrase
noun_phrase => Article Adj* noun
Adj* => Adj | Adj Adj*
verb_phrase => verb noun_phrase
Article =>  一个 | 这个
noun =>   女人 |  篮球 | 桌子 | 小猫
verb => 看着   |  坐在 |  听着 | 看见
Adj =>   蓝色的 |  好看的 | 小小的
"""


def get_generation_by_gram(grammar_str: str, target, stmt_split='=', or_split='|'):
    # 循环元字符串拼成{key: list}形式
    rules = {}
    for line in grammar_str.split('\n'):
        if line:
            stmt, expr = line.split(stmt_split)
            rules[stmt.strip()] = expr.split(or_split)

    return generate_n(rules, target=target)


def generate(grammar_rule, target):
    # 如果目标依然在字典里则进行递归
    if target in grammar_rule:
        candidates = grammar_rule[target]
        candidate = random.choice(candidates)
        return ''.join(generate(grammar_rule, target=c.strip()) for c in candidate.split())
    else:
        return target


# r = get_generation_by_gram(activity, "activity")
# print(r)
# r = get_generation_by_gram(activity_2, "activity")
# print(r)


def generate_n(grammar_rule, target, n=10):
    return [generate(grammar_rule, target) for _ in range(n)]


r = get_generation_by_gram(activity_2, "activity")

if "__main__" == __name__:
    print(r)

