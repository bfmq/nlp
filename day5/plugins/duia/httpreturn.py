#!/usr/bin/env python
# -*- coding:utf8 -*-
# __author__ = '北方姆Q'


def http_return(status=False, message=None, code=404):
    return {'status': status, 'message': message, 'code': code}
