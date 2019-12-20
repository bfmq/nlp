#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import re
import pinyin
import json
from functools import wraps
from functools import lru_cache
from collections import Counter, defaultdict


original_price = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 35]
price = defaultdict(int)
for i, p in enumerate(original_price):
    price[i+1] = p


def r(n):
    """
    假定跟中切割方式最大值获取都可以获取到
    则只需要比较本身长度价值与各种切割方式后的价值
    这里for i in range(1, n)应该还是可以优化成for i in range(1, n//2+1)
    因为 r(1)+r(n-1)与+r(n-1)+r(1)相同，没必要两次次计算，在n^n复杂度下/2还是比较有用的
    :param n: 木材长度
    :return:
    """
    return max([price[n]] + [r(i) + r(n-i) for i in range(1, n)])

print(r(10))
# print(r(30))


def memo(f):
    """
    定义already_computed记录缓存
    如果缓存命中直接返回
    缓存装饰器就是典型的空间换时间理念
    :param f:
    :return:
    """
    memo.already_computed = {}

    @wraps(f)
    def _wrap(arg):
        result = None

        if arg in memo.already_computed:
            result = memo.already_computed[arg]
        else:
            result = f(arg)
            memo.already_computed[arg] = result

        return result

    return _wrap


solution = {}


@memo
def r(n):
    """
    Args: n is the iron length
    Return: the max revenue
    """
    max_price, max_split = max(
        [(price[n], 0)] + [(r(i) + r(n - i), i) for i in range(1, n)], key=lambda x: x[0]
    )
    solution[n] = (n-max_split, max_split)
    return max_price


def parse_solution(n):
    """
    解析solution结果
    :param n: 总长度
    :return:
    """
    # 获取n长度分割左右节点
    left_split, right_split = solution[n]
    # 右节点为0则表示直接保留长度即可
    if right_split == 0: return [left_split]
    # 递归获取可继续分割的左右节点
    return parse_solution(left_split) + parse_solution(right_split)


print(r(20))
print(solution)
print(parse_solution(20))


@lru_cache(maxsize=2 ** 10)
def edit_distance(string1, string2):
    if len(string1) == 0: return len(string2)
    if len(string2) == 0: return len(string1)

    tail_s1 = string1[-1]       # 获取A串最后字符
    tail_s2 = string2[-1]       # 获取B串最后字符

    candidates = [
        # 给A减一位，编辑距离+1
        (edit_distance(string1[:-1], string2) + 1, 'DEL {}'.format(tail_s1)),
        # 给B加一位，编辑距离+1
        (edit_distance(string1, string2[:-1]) + 1, 'ADD {}'.format(tail_s2)),
    ]

    if tail_s1 == tail_s2:
        # 如果AB最后字符相同则向前推移一位继续比较，该位不需要计算编辑距离
        both_forward = (edit_distance(string1[:-1], string2[:-1]) + 0, '')
    else:
        # 否则则需要进行替换操作，该位编辑距离+1
        both_forward = (edit_distance(string1[:-1], string2[:-1]) + 1, 'SUB {} => {}'.format(tail_s1, tail_s2))

    # 增加做过的全部操作
    candidates.append(both_forward)

    # 全部操作中次数最小的就是可实现的最小编辑距离次数及操作
    min_distance, operation = min(candidates, key=lambda x: x[0])

    solution[(string1, string2)] = operation

    return min_distance


min_distance = edit_distance('ABCDE', 'ABCCEF')
print(min_distance)


chinese_dataset = '../day1/data/article_9k.txt'
CHINESE_CHARATERS = open(chinese_dataset, encoding='utf-8').read()
# r = pinyin.get('你好', format="strip", delimiter=" ")
# print(r)


def chinese_to_pinyin(character):
    return pinyin.get(character, format="strip", delimiter=" ")


def tokens(text):
    return re.findall('[a-z]+', text.lower())


# CHINESE_PINYIN_CORPYS = chinese_to_pinyin(CHINESE_CHARATERS)
# json.dump(tokens(CHINESE_PINYIN_CORPYS), open('./CHINESE_PINYIN.json', 'w', encoding='utf-8'))
# PINYIN_COUNT = Counter(tokens(CHINESE_PINYIN_CORPYS))
# json.dump(PINYIN_COUNT, open('./PINYIN_COUNT.json', 'w', encoding='utf-8'))
PINYIN_COUNT = json.load(open('./PINYIN_COUNT.json', encoding='utf-8'))


def correct(word):
    "Find the most possible pinyin based on edit distance."

    # Prefer edit distance 0, then 1, then 2; otherwise default to word itself.

    candidates = (known(edits0(word)) or
                  known(edits1(word)) or
                  known(edits2(word)) or
                  [word])
    return max(candidates, key=PINYIN_COUNT.get)


def known(words):
    "Return the pinyin we have noticed."
    return {w for w in words if w in PINYIN_COUNT}


def edits0(word):
    "Return all strings that are zero edits away from word (i.e., just word itself)."
    return {word}


def edits2(word):
    "Return all strings that are two edits away from this pinyin."
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


def edits1(word):
    "Return all strings that are one edit away from this pinyin."
    pairs      = splits(word)
    deletes    = [a+b[1:]           for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces   = [a+c+b[1:]         for (a, b) in pairs for c in alphabet if b]
    inserts    = [a+c+b             for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def splits(word):
    "Return a list of all possible (first, rest) pairs that comprise pinyin."
    return [(word[:i], word[i:])
            for i in range(len(word)+1)]

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def correct_sequence_pinyin(text_pingyin):
    return ' '.join(map(correct, text_pingyin.split()))


r1 = correct_sequence_pinyin('zhe sih yi ge ce sho')
print(r1)
r2 = correct_sequence_pinyin('wo xiang shagn qinng hua da xue')
print(r2)
