#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import json
import pprint
import os
import jieba
from collections import Counter
import pandas as pd


PINYIN_COUNT = json.load(open('./PINYIN_COUNT.json', encoding='utf-8'))
_2_gram_pinyin_counts = json.load(open('./2_gram_pinyin_counts.json', encoding='utf-8'))
alphabet = 'abcdefghijklmnopqrstuvwxyz'


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


def correct_sequence_pinyin(text_pingyin):
    return ' '.join(map(correct, text_pingyin.split()))


def cut(sentence):
    cut_list = splits(sentence)
    cut_list.sort(key=lambda x: PINYIN_COUNT.get(x[0], 0), reverse=True)
    left, right = cut_list[0]
    if not right: return [left]
    return [left] + cut(right)


def get_gram_count(word, wc):
    return wc[word] if word in wc else wc.most_common()[-1][-1]


def two_gram_model(sentence):
    tokens = cut(sentence)

    probability = 1

    for i in range(len(tokens)-1):
        word = tokens[i]
        next_word = tokens[i+1]

        _two_gram_c = get_gram_count(word + next_word, _2_gram_pinyin_counts)
        _one_gram_c = get_gram_count(next_word, PINYIN_COUNT)
        pro = _two_gram_c / _one_gram_c

        probability *= pro

    return probability


x = 'zehwomenquchifa'
c = cut(x)
csp = correct_sequence_pinyin(' '.join(c))
print(csp)
