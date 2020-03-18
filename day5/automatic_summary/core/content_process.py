#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import re
import pandas as pd


file_path = "../../data/export_sql_1558435/"
file_name = 'sqlResult_1558435.csv'
new_file_name = 'content.csv'

all_pd = pd.read_csv(file_path+file_name)
df = all_pd[['title', 'content']]
df = df.dropna()
df = df.drop_duplicates()


def strip_merge(content_lines):
    """
    经观察发现，可以用来切分的符号有
    \n
    \\n
    ?
    ;
    ■
    ↑
    *
    :param content_lines:
    :return:
    """
    content_lines_list = map(lambda content_line: content_line.strip(), re.split(r"[\n\\n;?■↑*□●]", content_lines))
    return ''.join(content_lines_list)

df['content'] = df['content'].apply(strip_merge)
df['title'] = df['title'].apply(strip_merge)
df = df.dropna()
df = df.drop_duplicates()
df.to_csv(file_path+new_file_name)
