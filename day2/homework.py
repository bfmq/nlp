#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import requests
from day2.class_code import bfs, geo_distance
from collections import defaultdict
from itertools import product
from bs4 import BeautifulSoup
from functools import reduce


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
    subway_new_connection = defaultdict(list)
    for k, v_list in subway_connection.items():
        for v in v_list:
            k_set = subway_lines_dict.get(k, set())
            v_set = subway_lines_dict.get(v, set())
            if k_set & v_set:
                subway_new_connection[k].append(v)

    return subway_new_connection

subway_new_connection = build_new_connection(subway_connection, subway_lines_dict)
r = bfs(subway_new_connection, '宋家庄', '望京')
# print(r)


# 但是实际情况中，我们不会只是为了到就可以
# 我们会选择坐的站点之间最短距离的、换成最少的、路过站最少的等等方式...
# 不过由于数据集过大bfs跑得太慢了，所以只写了代码逻辑....本质就是修改排序的逻辑


def bfs_by_distance(data, start, end):
    """
    按经历的所有地铁站之间距离总和最小计算路径
    :param data:
    :param start:
    :param end:
    :return:
    """
    def sort_by_distance(pathes):
        def get_distance_of_path(path):
            distance = 0
            for i, _ in enumerate(path[:-1]):
                distance += get_subway_distance(path[i], path[i + 1])
            return distance
        return sorted(pathes, key=get_distance_of_path)

    pathes = [[start]]      # 所有走过的路径列表

    while pathes:
        path = pathes.pop(0)    # 取最之前的第一个列表
        froniter = path[-1]     # 取该列表的最后一个元素节点城市作为需要被查询新路径的起点

        successsors = data[froniter]    # 获取该元素节点城市连接的城市列表
        for city in successsors:
            if city in path:    # 环检测，如果该城市连接的城市已经在本列表里则为环，没必要继续循环
                continue

            new_path = path + [city]    # 新的路径列表
            pathes.append(new_path)     # 更新所有走过的路径列表

        pathes = sort_by_distance(pathes)    # 将所有走过的路径列表按某规则排序
        print(pathes)
        if pathes and (end == pathes[0][-1]):   # 如果排序后的第一个列表的最后位就是目标则次列表就是符合规则的路径
            return pathes[0]

    return None


# r = bfs_by_distance(subway_connection, '宋家庄', '望京')
# print(r)


def bfs_by_len(data, start, end):
    """
    按经历的所有地铁站个数最少计算路径
    :param data:
    :param start:
    :param end:
    :return:
    """
    def sort_by_len(pathes):
        def get_len_of_path(path):
            return len(path)
        return sorted(pathes, key=get_len_of_path)

    pathes = [[start]]      # 所有走过的路径列表

    while pathes:
        path = pathes.pop(0)    # 取最之前的第一个列表
        froniter = path[-1]     # 取该列表的最后一个元素节点城市作为需要被查询新路径的起点

        successsors = data[froniter]    # 获取该元素节点城市连接的城市列表
        for city in successsors:
            if city in path:    # 环检测，如果该城市连接的城市已经在本列表里则为环，没必要继续循环
                continue

            new_path = path + [city]    # 新的路径列表
            pathes.append(new_path)     # 更新所有走过的路径列表

        pathes = sort_by_len(pathes)    # 将所有走过的路径列表按某规则排序
        print(pathes)
        if pathes and (end == pathes[0][-1]):   # 如果排序后的第一个列表的最后位就是目标则次列表就是符合规则的路径
            return pathes[0]

    return None

# r = bfs_by_len(subway_connection, '宋家庄', '望京')
# print(r)


def bfs_by_change(data, start, end):
    """
    按经历的换乘线站个数最少计算路径
    :param data:
    :param start:
    :param end:
    :return:
    """
    def sort_by_change(pathes):
        """
        如果某站跟它后两站所处地铁线无交集，则说明他们中间那站换乘了
        :param pathes:
        :return:
        """
        def get_change_of_path(path):
            change = 0
            for i, _ in enumerate(path[:-2]):
                if not subway_lines_dict.get(path[i], set()) & subway_lines_dict.get(path[i+2], set()):
                    change += 1
            return change
        return sorted(pathes, key=get_change_of_path)

    pathes = [[start]]      # 所有走过的路径列表

    while pathes:
        path = pathes.pop(0)    # 取最之前的第一个列表
        froniter = path[-1]     # 取该列表的最后一个元素节点城市作为需要被查询新路径的起点

        successsors = data[froniter]    # 获取该元素节点城市连接的城市列表
        for city in successsors:
            if city in path:    # 环检测，如果该城市连接的城市已经在本列表里则为环，没必要继续循环
                continue

            new_path = path + [city]    # 新的路径列表
            pathes.append(new_path)     # 更新所有走过的路径列表

        pathes = sort_by_change(pathes)    # 将所有走过的路径列表按某规则排序
        print(pathes)
        if pathes and (end == pathes[0][-1]):   # 如果排序后的第一个列表的最后位就是目标则次列表就是符合规则的路径
            return pathes[0]

    return None

# r = bfs_by_change(subway_connection, '宋家庄', '望京')
# print(r)
