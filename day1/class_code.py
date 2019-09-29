#!/usr/bin/env python
# -*- coding:utf8 -*-
# __author__ = '北方姆Q'
# __datetime__ = 2019/9/29 10:32


import os
import random
import jieba
from collections import Counter


hello_rules = """
say_hello = names hello tail 
names = name names | name
name = Jhon | Mike | 老梁 | 老刘 
hello = 你好 | 您来啦 | 快请进
tail = 呀 | ！
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

simple_programming = """
if_stmt => if ( cond ) { stmt }
cond => var op var
op => | == | < | >= | <= 
stmt => assign | if_stmt
assign => var = var
var =>  char var | char
char => a | b |  c | d | 0 | 1 | 2 | 3
"""


def get_generation_by_gram(grammar_str: str, target, stmt_split='=', or_split='|'):
    # 循环元字符串拼成{key: list}形式
    rules = {}
    for line in grammar_str.split('\n'):
        if line:
            stmt, expr = line.split(stmt_split)
            rules[stmt.strip()] = expr.split(or_split)

    generated = generate(rules, target=target)
    return generated


def generate(grammar_rule, target):
    # 如果目标依然在字典里则进行递归
    if target in grammar_rule:
        candidates = grammar_rule[target]
        candidate = random.choice(candidates)
        return ''.join(generate(grammar_rule, target=c.strip()) for c in candidate.split())
    else:
        return target


r = get_generation_by_gram(hello_rules, target='say_hello', stmt_split='=')
print(r)

r = get_generation_by_gram(simple_grammar, target='sentence', stmt_split='=>')
print(r)

for i in range(30):
    r = get_generation_by_gram(simple_programming, target='if_stmt', stmt_split='=>')
    print(r)


###############################################################


BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = f"{BaseDir}/day1/data/article_9k.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    FILE = f.read()


def cut(string):
    return list(jieba.cut(string))


TOKENS = cut(FILE[:1000000])
words_count = Counter(TOKENS)
_2_gram_words = [
    TOKENS[i] + TOKENS[i+1] for i in range(len(TOKENS)-1)
]
_2_gram_word_counts = Counter(_2_gram_words)


def get_gram_count(word, wc):
    """
    获取字符串在总字符表中的次数
    :param word: 需要查询的字符串
    :param wc: 总字符表
    :return: 该字符串出现的次数，如没有则定为出现最少次数字符串的次数
    """
    return wc[word] if word in wc else wc.most_common()[-1][-1]


def two_gram_model(sentence):
    """
    分别计算句子中该单词在总字符表中出现的次数
    该单词跟后一单词在二连总字符表中出现的次数
    做比后的连续乘积
    :param sentence:  需要验证的句子
    :return:
    """
    tokens = cut(sentence)

    probability = 1

    for i in range(len(tokens)-1):
        word = tokens[i]
        next_word = tokens[i+1]

        _two_gram_c = get_gram_count(word + next_word, _2_gram_word_counts)
        _one_gram_c = get_gram_count(next_word, words_count)
        pro = _two_gram_c / _one_gram_c

        probability *= pro

    return probability


r = two_gram_model("这个花特别好看")
print(r)
r = two_gram_model("花这个特别好看")
print(r)
r = two_gram_model("自然语言处理")
print(r)
r = two_gram_model("处语言理自然")
print(r)
r = two_gram_model("前天早上")
print(r)
