#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import re
import pandas as pd


file_path = "./data/"
file_name = 'sqlResult_1558435.csv'     # sqlResult_1558435太大了，处理完就删除了
new_file_name = 'content.csv'

all_pd = pd.read_csv(file_path+file_name)
df = all_pd[['source', 'content']]
df = df.dropna()
df = df.drop_duplicates()


def strip_merge(content_lines):
    content_lines_list = map(lambda content_line: content_line.strip(), re.split(r"[\n\\n;?■↑*□●]", content_lines))
    return ''.join(content_lines_list)


def set_source(source_lines):
    """
    如果source是新华社则设置为1，不是则为0
    :param source_lines:
    :return:
    """
    return 1 if '新华社' in source_lines else 0

df['content'] = df['content'].apply(strip_merge)
df['source'] = df['source'].apply(set_source)

df = df.dropna()
df = df.drop_duplicates()
df.to_csv(file_path+new_file_name)
