#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import jieba, os, re


def get_stopwords():
    stopword_set = set()
    with open("../../data/stop/stopwords.txt", 'r', encoding="utf-8") as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip("\n"))
    return stopword_set


def parse_zhwiki(read_file_path, save_file_path):
    regex_str = "[^<doc.*>$]|[^</doc>$]"
    file = open(read_file_path, "r", encoding="utf-8")
    output = open(save_file_path, "w+", encoding="utf-8")
    content_line = file.readline()
    stopwords = get_stopwords()
    article_contents = ""
    while content_line:
        match_obj = re.match(regex_str, content_line)
        content_line = content_line.strip("\n")
        if len(content_line) > 0:
            if match_obj:
                # 使用jieba进行分词
                words = jieba.cut(content_line, cut_all=False)
                for word in words:
                    if word not in stopwords:
                        article_contents += word + " "
            else:
                if len(article_contents) > 0:
                    output.write(article_contents + "\n")
                    article_contents = ""
        content_line = file.readline()
    output.close()


def generate_corpus():
    save_path = zhwiki_path = "../../data/wiki/wiki"
    for i in range(3):
        file_path = os.path.join(zhwiki_path, str("zh_wiki_0%s_jt" % str(i)))
        parse_zhwiki(file_path, os.path.join(save_path, "wiki_corpus0%s" % str(i)))

