#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import CrossEntropyLoss


data_path = './data/'

train_data1 = data_path + 'd1.txt'
train_data2 = data_path + 'd3.txt'
train_data3 = data_path + 'd3.txt'


def read_data(file):
    with open(file,'r',encoding='utf8') as data:
        lines = [l.strip() for l in data]
    dials = []
    for l in lines:
        if l == '<dial>':
            dial = []
        elif l == '</dial>':
            dials += [dial]
        else:
            dial += [l.strip()]
    return dials


dials = read_data(train_data1) + read_data(train_data2) + read_data(train_data3)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
weights = torch.load('../Data_source/medium_ft.pkl')
medium_config = GPT2Config(n_embd = 1024,n_layer = 24, n_head = 16)
model = GPT2LMHeadModel(medium_config)

weights['lm_head.weight'] = weights['lm_head.decoder.weight']
weights.pop('lm_head.decoder.weight',None)

model.load_state_dict(weights)
model.train()


class InputFeature(object):
    def __init__(self, input_ids, position_ids, token_type_ids,
                 lm_labels=None, input_len=None):

        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.lm_lanels = lm_labels
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len


class GPT2Dataset(Dataset):
    def __init__(self, dials, max_len=1024):
        self.max_len = max_len
        self.features = self.build_input_feature(dials)

    def __getitem__(self, i):
        feat_dict = self.features[i]
        if self.max_len is not None and feat_dict['input_len'] > self.max_len:
            feat_dict['input_ids'] = feat_dict['input_dis'][-self.max_len:]
            feat_dict['position_ids'] = feat_dict['position_ids'][-self.max_len:]
            feat_dict['token_type_ids'] = feat_dict['token_type_ids'][-self.max_len:]
            feat_dict['lm_labels'] = feat_dict['lm_labels'][-self.max_len:]
        feat = InputFeaturet(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def build_input_feature(dials, end_text='<|endoftext|>'):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        feature = []
        for dial in dials:
            inputs = sum([tokenizer.encode(u) for u in dial[:-1]], [])
            lm_labels = [-1] * len(inputs) + tokenizer.encode(dial[-1] + end_text)
            token_type_ids = [0] * len(inputs) + [1.0] * (len(tokenizer.encode(dial[-1] + end_text)))
            input_ids = inputs + tokenizer.encode(end_text + dial[-1])
            input_len = len(input_ids)
            position_ids = list(range(len(input_ids)))

            feat_dict = {'input_ids': input_ids,
                         'position_ids': position_ids,
                         'token_type_ids': token_type_ids,
                         'lm_labels': lm_labels,
                         'input_len': input_len}
            feature.append(feat_dict)
        return feature

    @staticmethod
    def collate(features):
        inputs_ids = pad_sequence([torch.tensor(f['input_ids'], dtype=torch.long)
                                   for f in features], batch_first=True, padding_value=0)

        position_ids = pad_sequence([torch.tensor(f['position_ids'], dtype=torch.long)
                                     for f in features], batch_first=True, padding_value=0)

        token_type_ids = pad_sequence([torch.tensor(f['token_type_ids'], dtype=torch.long)
                                       for f in features], batch_first=True, padding_value=0)

        labels = pad_sequence([torch.tensor(f['lm_labels'], dtype=torch.long)
                               for f in features], batch_first=True, padding_value=-1)

        return (inputs_ids, position_ids, token_type_ids, labels)


dataset = GPT2Dataset.build_input_feature(dials)
loader = DataLoader(dataset,collate_fn=GPT2Dataset.collate,batch_size=1)


def run(model, train_dataloader, learning_rate, epoches):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    step = 0
    epoch = 0

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    loss_function = CrossEntropyLoss(ignore_index=-1, reduction='mean')
    while epoch < epoches:
        running_loss = 0
        try:
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader)) as pbar:
                for i, batch in pbar:
                    input_ids, position_ids, token_type_ids, label_ids = batch
                    logits = model(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
                    lm_logits = logits[0]

                    loss = loss_function(lm_logits.view(-1, lm_logits.size(-1)), label_ids.view(-1))
                    running_loss += loss.item()
                    pbar.set_description('Train (Epoch{}):{:.4f}'.format(epoch, running_loss / (step + 1)))
                    optimizer.zero_grad()  # dw = 0
                    loss.backward()
                    optimizer.step()  # w = w +dw
                    step += 1
                epoch += 1
        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()
    torch.save({'model': model.state_dict(),
                'epoch': epoch})

run(model,loader,1e-4,10)
