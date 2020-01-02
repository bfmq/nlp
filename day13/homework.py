#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader,Dataset


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
weights = torch.load('../data/small_ft.pkl')
medium_config = GPT2Config(n_embd = 768,n_layer = 12, n_head = 12)
model = GPT2LMHeadModel(medium_config)

weights['lm_head.weight'] = weights['lm_head.decoder.weight']
weights.pop('lm_head.decoder.weight',None)

model.load_state_dict(weights)
model.train()








