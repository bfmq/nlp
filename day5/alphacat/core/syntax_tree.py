#!/usr/bin/env python
# -*- coding:utf8 -*-
# __author__ = '北方姆Q'

import random


polite_language = """
polite = 人物 时间 语气词 动作 事件 结束标点 | 其他 结束标点
人物 = 我 | 我们 | 在下 | 后台 | 机器人
时间 = 现在 | 目前 | 此时此刻 | 现阶段 | 当下
语气词 = 还 | 可能 | 也许
动作 = 无法解决 | 不清楚 | 不明白 | 还在学习 | 无法回答 | 还不会
事件 = 该问题 | 这个提问
结束标点 = 。 | ... | ~ | ~~~
其他 = 学无止境 | 看来还有很多需要学习的知识啊 | 被问住了 | emm，需要再回去翻翻书 | 不知道您在说些什么
"""


def get_generation_by_gram(grammar_str=polite_language, target='polite', stmt_split='=', or_split='|'):
    # 循环元字符串拼成{key: list}形式
    rules = {}
    for line in grammar_str.split('\n'):
        if line:
            stmt, expr = line.split(stmt_split)
            rules[stmt.strip()] = expr.split(or_split)

    return generate(rules, target=target)


def generate(grammar_rule, target):
    # 如果目标依然在字典里则进行递归
    if target in grammar_rule:
        candidates = grammar_rule[target]
        candidate = random.choice(candidates)
        return ''.join(generate(grammar_rule, target=c.strip()) for c in candidate.split())
    else:
        return target
