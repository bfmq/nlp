#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.autograd import Variable
import math


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)

    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                      diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


a = torch.tensor([[1,2,3,0,0,0]])
b = torch.tensor([[1,2,3,4,0,0]])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0., max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-2, -1))  # q*k.T

        if scale:
            attention = attention * scale

        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)

        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = d_model // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_q = nn.Linear(d_model, d_model)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(d_model, d_model)  # 全链接层
        self.norm = nn.LayerNorm(d_model)

    def forward(self, keys, values, queries, attn_mask=None):
        residul = queries  # residual connection
        batch_size = keys.size(0)

        keys = self.linear_k(keys)
        values = self.linear_v(values)
        queries = self.linear_q(queries)

        keys = keys.view(batch_szie, -1, self.num_heads, self.dim_per_head).transpose(1,
                                                                                      2)  # [batch_size, num_heads,length,d_model]
        values = values.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        queries = queries.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        scale = (keys.size(-1)) ** -0.5

        context = self.dot_product_attention(queries, keys, values, scale, attn_mask)

        context = context.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.num_heads * self.dim_per_head)

        return self.norm(residul + self.linear_final(context))


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.w2(F.relu(self.w1(x)))
        return self.norm(x + self.dropout(output))  # residual connection


class Encoder_Layer(nn.Module):  # one block

    def __init__(self, d_model=512, num_heads=8,
                 ffn_dim=2048, dropout=0.0):
        super(Encoder_Layer, self).__init__()
        self.attention = MultiHeadAttention(d_mode, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, ffn_dim, dropout)

    def forward(self, x, attn_mask=None):
        context = self.attention(x, x, x, attn_mask)
        output = self.feed_forward(context)
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len,
                 num_layers=6, d_model=512, num_heads=8,
                 ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, ffn_dim, dropout)
                                             for _ in range(num_layers)])

        self.norm = nn.LayerNorm(d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len, dropout)

    def forward(self, x, seq_embedding):
        embedding = seq_embedding(x)
        output = self.pos_embedding(embedding)

        self_attention_mask = padding_mask(x.x)

        for encoder in self.encoder_layers:
            output = encoder(output, self_attention_mask)

        return self.norm(output)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8,
                 ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):
        dec_output = self.attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        dec_output = self.attention(enc_outputs, enc_outputs, dec_output, context_attn_mask)

        dec_output = self.feed_forward(dec_output)

        return dec_output


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6,
                 d_model=512, num_heads=8, ffn_dim=2048,
                 dropout=0.0):
        super(Decoder, self).__init__()
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, inputs, enc_out, seq_embedding, context_attn_mask=None):
        embedding = seq_embedding(inputs)
        output = embedding + self.pos_embedding(embedding)

        self_attention_padding_mask = padding_max(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        for decoder in self.decoder_layers:
            output = decoder(output, enc_out, self_attn_mask, context_attn_mask)

        output = self.linear(output)
        return output


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len,
                 num_layers=6,
                 stack_layers=6,
                 d_model=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(vocab_size, max_len, num_layers, d_model, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(vocab_size, max_len, num_layers, d_model, num_heads, ffn_dim, dropout)

        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.linear = nn.Linear(d_model,vocab_size,bias=False)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, dec_tgt):
        context_attn_mask_dec = padding_mask(dec_tgt, src_seq)

        en_output = self.encoder(src_seq, self.embedding)
        dec_output = self.decoder(dec_tgt, en_output, self.embedding, context_attn_mask_dec)

        return dec_output








