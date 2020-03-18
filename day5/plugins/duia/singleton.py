#!/usr/bin/env python
# -*- coding:utf8 -*-
# __author__ = '北方姆Q'

import threading


class Singleton(object):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with cls._instance_lock:
                cls._instance = object.__new__(cls)
        return cls._instance
