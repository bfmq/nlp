#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-

import numpy as np

sentence_list = ['充分提升部队快速响应、快速出动和快速部署的能力', '央视网消息','近日','陆军第79集团军组织部队开展实战条件下的战备拉动演练', '通过不断简化指挥程序、优化出动流程',]
sub_sentences = ['央视网消息','近日','陆军第79集团军组织部队开展实战条件下的战备拉动演练', '充分提升部队快速响应、快速出动和快速部署的能力', '通过不断简化指挥程序、优化出动流程']

summarized = np.zeros_like(sub_sentences)

current_sen = 0  # 当前长度
max_count = 20

for sentence in sentence_list:
    if current_sen < max_count:
        current_sen += len(sentence)
        index = sub_sentences.index(sentence)
        summarized[index] = sub_sentences[index]
    else:
        break

summarized.append(100)
print(summarized)
