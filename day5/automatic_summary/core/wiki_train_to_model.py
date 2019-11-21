#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')

logging.root.setLevel(level=logging.INFO)

logging.info("running %s" % ' '.join(sys.argv))

# if len(sys.argv) < 4:
#     print(globals()['__doc__'] % local())
#     sys.exit(1)

inp, outp1, outp2 = sys.argv[1:4]

model = Word2Vec(LineSentence(inp), size = 512, window = 10, min_count = 5, workers = multiprocessing.cpu_count())

model.save(outp1)

model.wv.save_word2vec_format(outp2)
