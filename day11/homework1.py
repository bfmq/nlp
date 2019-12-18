#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

from functools import lru_cache


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


def parse_solution(string1, string2):
    """
    解析solution结果
    :param n: 总长度
    :return:
    """
    if string1 == string2: return []
    if len(string1) == 0: return ['ADD {}'.format(i) for i in string2]
    if len(string2) == 0: return ['DEL {}'.format(i) for i in string1]

    operation = solution.get((string1, string2))

    if operation == '':
        string1, right_str = string1[:-1], string2[:-1]
        return parse_solution(string1, right_str)
    elif 'ADD' in operation:
        string2 = string2[:-1]
    elif 'DEL' in operation:
        string1 = string1[:-1]
    else:
        string1, right_str = string1[:-1], string2[:-1]

    return parse_solution(string1, string2)+[operation]

solution = {}
s1 = 'DVDFUI'
s2 = 'VBRTWUIBIUWERT'
min_distance = edit_distance(s1, s2)
print(min_distance)
print(parse_solution(s1, s2))
