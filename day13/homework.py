#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
weights = torch.load('./data/small_ft.pkl')
medium_config = GPT2Config(n_embd=768, n_layer=12, n_head=12)
model = GPT2LMHeadModel(medium_config)
weights['lm_head.weight'] = weights['lm_head.decoder.weight']
weights.pop('lm_head.decoder.weight', None)
model.load_state_dict(weights)
model.train()
model.eval()


def sentences_generater(text):
    sentences_list = [text]
    sentence = ''
    count = 0

    while sentence not in sentences_list and count < 10:
        indexed_tokens = tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        predicted_text = tokenizer.decode(predicted_index)

        if predicted_text == '<|endoftext|>':
            if sentence:
                sentences_list.append(sentence)
                sentence = ''
                count += 1
        else:
            sentence += predicted_text

        text = text + predicted_text

    return sentences_list


questions = ['Does money buy happiness ?',
             'What is the best way to buy happiness?',
             'what is the meaning of a godd life ?',
             'How to be a good person ?',
             'what do you think nlp？']

for question in questions:
    sentences = sentences_generater(question)
    for sentence in sentences:
        print(sentence)
    print('=============')

