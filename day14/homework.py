#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from day1.homework2 import two_gram_model
from itertools import chain


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
weights = torch.load('../day13/data/small_ft.pkl')
medium_config = GPT2Config(n_embd=768, n_layer=12, n_head=12)
model = GPT2LMHeadModel(medium_config)
weights['lm_head.weight'] = weights['lm_head.decoder.weight']
weights.pop('lm_head.decoder.weight', None)
model.load_state_dict(weights)
model.train()
model.eval()


def predict(text, k=10):
    """
    对文本进行下一次的预测
    :param text: 文本
    :param k: 预测值的个数
    :return: [文本+预测1，文本+预测2，...，文本+预测10]
    """
    new_text = []
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # 将argmax改成topk即可获取多个预测值
    topk = torch.topk(predictions[0, -1, :], k)
    for token in topk.indices.cpu().numpy():
        new_text.append(text+tokenizer.decode([token]))
    return new_text


def keep_k(texts, k):
    """
    保证总列表中保留前k合理的句子，分数使用two_gram_model，可以改
    :param texts: 文本列表
    :param k: 保留个数
    :return: [长度k的文本列表]
    """
    texts = sorted(texts, key=lambda x: two_gram_model(x), reverse=True)[:k]
    return texts


def sentences_generater(text):
    count = 0
    result_list = set([text])

    while count < 10:
        # 每次将所有结果进行预测，等于出现长度为100的列表
        result_list = set(chain(*[predict(q, 10) for q in result_list]))
        # 在保留前10个恢复长度
        result_list = keep_k(result_list, 10)
        count += 1

    return result_list[random.randrange(10)].split(text)[-1]


questions = ['Does money buy happiness ?',
             'What is the best way to buy happiness?',
             'what is the meaning of a godd life ?',
             'How to be a good person ?',
             'what do you think nlp？']

# 跑起来还是比较慢，是一个缺点
# 输出结果一般，应该是跟打分函数情况有关系
for question in questions:
    answer = sentences_generater(question)
    print(answer)
    print('=============')



