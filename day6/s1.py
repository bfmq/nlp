#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

import numpy as np
inputs = np.linspace(-2*np.pi, 2*np.pi, 10)[:, None]
outputs = np.sin(inputs)
outputs2 = np.random.normal(size=10)
print(outputs)
print(outputs2)