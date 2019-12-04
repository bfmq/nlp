#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import jieba, os, re
from gensim.corpora import WikiCorpus


def get_wiki_text():
    outp = "../../data/wiki/wiki.zh.txt"
    inp = "../../data/wiki/zhwiki-20190720-pages-articles-multistream.xml.bz2"

    space = " "

    output = open(outp, 'w', encoding='utf-8')

    # gensim里的维基百科处理类WikiCorpus
    wiki = WikiCorpus(inp, lemmatize=False, dictionary=[])

    # 通过get_texts将维基里的每篇文章转换位1行text文本，并且去掉了标点符号等内容
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
    output.close()


def remove_words():
    output = open('data/wiki.zh.txt', 'w', encoding='utf-8')
    inp = open('data/wiki.zh.zh.txt', 'r', encoding='utf-8')

    for line in inp.readlines():
        ss = re.findall('[\n\s*\r\u4e00-\u9fa5]', line)
        output.write("".join(ss))


def separate_words():
    output = open('data/wiki.corpus.txt', 'w', encoding='utf-8')
    inp = open('data/wiki.zh.txt', 'r', encoding='utf-8')

    for line in inp.readlines():
        seg_list = jieba.cut(line.strip())
        output.write(' '.join(seg_list) + '\n')

# get_wiki_text()
# remove_words()
# separate_words()

