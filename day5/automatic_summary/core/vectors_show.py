#!/usr/bin/env python
# __author__ = '北方姆Q'
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from gensim.models import word2vec
from sklearn.manifold import TSNE


model = word2vec.Word2Vec.load("../../models/wiki/wiki_corpus.model")


def tsne_plot(model):
    labels = []
    tokens = []

    i = 0
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
        i += 1

        if i > 50:
            break

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


tsne_plot(model)
