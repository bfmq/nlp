#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import requests
import json
from day2.class_code import bfs, geo_distance
from collections import defaultdict
from itertools import product
from bs4 import BeautifulSoup


with open('data', 'r', encoding='utf-8') as f:
    file = f.read()


subway_info = {subway.split(',')[0]: (float(subway.split(',')[1]), float(subway.split(',')[2])) for subway in file.split('|')}


def get_subway_distance(subway1, subway2):
    return geo_distance(subway_info[subway1], subway_info[subway2])

r = get_subway_distance('宋家庄', '刘家窑')
# print(r)


def build_connection(subway_info):
    """
    构造距离为3的地铁站信息
    :param subway_info:
    :return:
    """
    subways_connection = defaultdict(list)
    subways = list(subway_info.keys())
    for c1, c2 in product(subways, subways):
        if c1 == c2:
            continue

        if get_subway_distance(c1, c2) < 3:
            subways_connection[c1].append(c2)
    return subways_connection


subway_connection = build_connection(subway_info)
r = bfs(subway_connection, '宋家庄', '望京')
# print(r)

# 出现问题的原因在于从坐标构建的字典默认所有地铁站都是可以相互连接的
# 但是实际情况肯定不是，地铁站只能连接到自己所处的线站
# 因此核心问题在于subway_connection这个字典并不科学，所欲数据的预处理没做完全
# 只要subway_connection中每个地铁站连接的是自己所处的线站即可，对此我们还需要一个地铁线沿站的数据


def make_subway_lines():
    """
    爬取页面构造 地铁站与地铁线的关系
    :return: {xxx站：set(A号线，B号线)}
    """
    subway_lines_dict = defaultdict(set)
    url = 'http://bj.bendibao.com/ditie/linemap.shtml'
    soup = BeautifulSoup(requests.get(url).text, 'lxml')
    line_lists = soup.select('div[class="line-list"]')
    for line_list in line_lists:
        line_name = line_list.select_one('a').get_text()
        line_states = line_list.select('div[class="station"]')
        for line_state in line_states:
            line_state_name = line_state.select_one('a[class="link"]').get_text()
            subway_lines_dict[line_state_name].add(line_name)

    return subway_lines_dict


subway_lines_dict = make_subway_lines()


def build_new_connection(subway_connection, subway_lines_dict):
    """
    循环之前的连接信息，如果本站点与连接站点没有交集，则说明不在一条线上，则删除
    :param subway_connection: 之前的连接信息
    :param subway_lines_dict:  地铁站与地铁线的关系
    :return:
    """
    for subway_line in subway_connection:
        for subway_connection_line in subway_connection.get(subway_line):
            if not subway_lines_dict.get(subway_connection_line, set()) & subway_lines_dict.get(subway_line, set()):
                subway_connection.get(subway_line).remove(subway_connection_line)

    return subway_connection


subway_connection = build_new_connection(subway_connection, subway_lines_dict)
# for i in subway_connection:
#     print(f"{i}---{subway_connection[i]}")
r = bfs(subway_connection, '宋家庄', '望京')
print(r)
