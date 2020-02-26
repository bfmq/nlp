#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup


def spider_man(contents):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3741.400 QQBrowser/10.5.3863.400',
    }

    url = 'https://baike.baidu.com/item/' + contents
    response = requests.get(url, headers=headers)

    if response.request.url == 'https://baike.baidu.com/error.html':
        # 百度百科返回值永远是200，如果不在百科内它第一跳转页是error.html，只能用这个判断
        return False
    else:
        # 有百科的话获取基本描述即可
        soup = BeautifulSoup(response.content.decode(), 'html.parser')
        label = soup.head.find('meta', attrs={"name": "description"})
        return f'{label.get("content")}' \
               f'<a href={response.request.url} target="_blank">点此跳转查看详细页面</a>'
