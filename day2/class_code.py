#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import re
import random
import math
import networkx as nx
from sklearn.datasets import load_boston
from collections import defaultdict
from itertools import product


coordination_source = """
{name:'兰州', geoCoord:[103.73, 36.03]},
{name:'嘉峪关', geoCoord:[98.17, 39.47]},
{name:'西宁', geoCoord:[101.74, 36.56]},
{name:'成都', geoCoord:[104.06, 30.67]},
{name:'石家庄', geoCoord:[114.48, 38.03]},
{name:'拉萨', geoCoord:[102.73, 25.04]},
{name:'贵阳', geoCoord:[106.71, 26.57]},
{name:'武汉', geoCoord:[114.31, 30.52]},
{name:'郑州', geoCoord:[113.65, 34.76]},
{name:'济南', geoCoord:[117, 36.65]},
{name:'南京', geoCoord:[118.78, 32.04]},
{name:'合肥', geoCoord:[117.27, 31.86]},
{name:'杭州', geoCoord:[120.19, 30.26]},
{name:'南昌', geoCoord:[115.89, 28.68]},
{name:'福州', geoCoord:[119.3, 26.08]},
{name:'广州', geoCoord:[113.23, 23.16]},
{name:'长沙', geoCoord:[113, 28.21]},
//{name:'海口', geoCoord:[110.35, 20.02]},
{name:'沈阳', geoCoord:[123.38, 41.8]},
{name:'长春', geoCoord:[125.35, 43.88]},
{name:'哈尔滨', geoCoord:[126.63, 45.75]},
{name:'太原', geoCoord:[112.53, 37.87]},
{name:'西安', geoCoord:[108.95, 34.27]},
//{name:'台湾', geoCoord:[121.30, 25.03]},
{name:'北京', geoCoord:[116.46, 39.92]},
{name:'上海', geoCoord:[121.48, 31.22]},
{name:'重庆', geoCoord:[106.54, 29.59]},
{name:'天津', geoCoord:[117.2, 39.13]},
{name:'呼和浩特', geoCoord:[111.65, 40.82]},
{name:'南宁', geoCoord:[108.33, 22.84]},
//{name:'西藏', geoCoord:[91.11, 29.97]},
{name:'银川', geoCoord:[106.27, 38.47]},
{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
{name:'香港', geoCoord:[114.17, 22.28]},
{name:'澳门', geoCoord:[113.54, 22.19]}
"""


def get_city_info(data):
    """
    从字符串解析出城市信息
    :param data:  源数据
    :return:  {(str: (float, float), (str: (float, float)}
    """
    city_dict = {}

    for line in data.split('\n'):
        if line.strip() == "":
            continue

        if line.startswith("//"):
            line.strip('//')

        # 匹配城市名
        city_name = re.findall("name:'(\w+)'", line)[0]
        # 匹配城市坐标
        x_y = re.findall("Coord:\[(\d+.\d+),\s(\d+.\d+)\]", line)[0]
        x_y = tuple(map(float, x_y))
        city_dict[city_name] = x_y

    return city_dict

city_info = get_city_info(coordination_source)


def geo_distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    # >>> origin = (48.1372, 11.5756)  # Munich
    # >>> destination = (52.5186, 13.4083)  # Berlin
    # >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


def get_city_distance(city1, city2):
    return geo_distance(city_info[city1], city_info[city2])

r = get_city_distance("澳门", "拉萨")
# print(r)


def build_connection(city_info):
    """
    生成距离700以内的城市对应列表
    :param city_info: 城市地理信息
    :return: {str: [], str: [], }
    """
    cities_connection = defaultdict(list)
    cities = list(city_info.keys())
    for c1, c2 in product(cities, cities):
        if c1 == c2:
            continue

        if get_city_distance(c1, c2) < 700:
            cities_connection[c1].append(c2)
    return cities_connection


cities_connection = build_connection(city_info)
cities_connection_graph = nx.Graph(cities_connection)
nx.draw(cities_connection_graph,city_info,with_labels=True,node_size=10)


def bfs(data, start, end):
    """
    广度优先查找
    :param data:  源数据
    :param start:  起始点
    :param end:  结束点
    :return:
    """
    pathes = [[start]]      # 所有走过的路径列表
    visited = set()         # 已查找点集合

    while pathes:
        path = pathes.pop(0)    # 取最之前的第一个列表
        froniter = path[-1]     # 取该列表的最后一个元素节点城市作为需要被查询新路径的起点

        if froniter in visited:     # 如果该元素节点城市已查则跳过
            continue

        successsors = data[froniter]    # 获取该元素节点城市连接的城市列表
        for city in successsors:
            if city in path:    # 环检测，如果该城市连接的城市已经在本列表里则为环，没必要继续循环
                continue

            new_path = path + [city]    # 新的路径列表

            # 跟dfs有更好的对比可以用 pathes = pathes + [new_path]
            pathes.append(new_path)     # 更新所有走过的路径列表

            if city == end:
                return new_path

        visited.add(froniter)           # 设置该节点已查，提升效率

    return None

r = bfs(cities_connection, "嘉峪关", "哈尔滨")
# print(r)


def dfs(data, start, end):
    """
    深度优先查找
    :param data:  源数据
    :param start:  起始点
    :param end:  结束点
    :return:
    """
    pathes = [[start]]
    visited = set()

    while pathes:
        path = pathes.pop(0)
        froniter = path[-1]

        if froniter in visited:
            continue

        successsors = data[froniter]
        for city in successsors:
            if city in path:
                continue

            new_path = path + [city]
            pathes = [new_path] + pathes

            if city == end:
                return new_path

        visited.add(froniter)

    return None

r = dfs(cities_connection, "嘉峪关", "哈尔滨")
# print(r)


def bfs_2(data, start, end, search_strategy):
    """
    广度优先查找的变化方式
    :param data:  源数据
    :param start:  起始点
    :param end:  结束点
    :return:
    """
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

        pathes = search_strategy(pathes)    # 将所有走过的路径列表按某规则排序

        if pathes and (end == pathes[0][-1]):   # 如果排序后的第一个列表的最后位就是目标则次列表就是符合规则的路径
            return pathes[0]

    return None


def sort_by_distance(pathes):
    def get_distance_of_path(path):
        distance = 0
        for i, _ in enumerate(path[:-1]):
            distance += get_city_distance(path[i], path[i+1])
        return distance
    return sorted(pathes, key=get_distance_of_path)

r = bfs_2(cities_connection, "北京", "上海", search_strategy=sort_by_distance)
# print(r)

##################################################################

data_set = load_boston()
x, y = data_set['data'], data_set['target']
# print(x.shape)
# print(y.shape)
# print(x[0])
# print(data_set.feature_names)
X_rm = x[:,5]


def price(rm, k, b):
    return k * rm + b


def loss(y, y_hat):
    return sum((y_i - y_hat_i)**2 for y_i, y_hat_i in zip(list(y), list(y_hat)))/len(list(y))


def partial_derivative_k(x, y, y_hat):
    n = len(y)
    gradient = 0
    for x_i, y_i, y_hat_i in zip(list(x), list(y), list(y_hat)):
        gradient += (y_i-y_hat_i) * x_i
    return -2/n * gradient


def partial_derivative_b(y, y_hat):
    n = len(y)
    gradient = 0
    for y_i, y_hat_i in zip(list(y), list(y_hat)):
        gradient += (y_i-y_hat_i)
    return -2 / n * gradient


def boston_loss():
    k = random.random() * 200 - 100
    b = random.random() * 200 - 100

    learning_rate = 1e-3

    iteration_num = 200
    losses = []
    for i in range(iteration_num):
        price_use_current_parameters = [price(r, k, b) for r in X_rm]  # \hat{y}

        current_loss = loss(y, price_use_current_parameters)
        losses.append(current_loss)
        print("Iteration {}, the loss is {}, parameters k is {} and b is {}".format(i, current_loss, k, b))

        k_gradient = partial_derivative_k(X_rm, y, price_use_current_parameters)
        b_gradient = partial_derivative_b(y, price_use_current_parameters)

        k = k + (-1 * k_gradient) * learning_rate
        b = b + (-1 * b_gradient) * learning_rate
    best_k = k
    best_b = b

# boston_loss()
