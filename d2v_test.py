# -*- coding: utf-8 -*-
from gensim import models
from datetime import datetime
from gensim.models.doc2vec import LabeledSentence
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

INPUT_DOC_DIR = r'C:\test\word2vec\text'
OUTPUT_MODEL = r'C:\test\IBM\giji\doc2vec_s400m1i10.model'
model = models.Doc2Vec.load(OUTPUT_MODEL)

query = unicode('【NPD様】第2回セッション打ち合わせ_議事録.xlsx.議事録.txt',"utf-8")

print "the query file name:", query
for i in model.docvecs.most_similar(query, topn=10):
    print i[0], i[1]

keyword = "設計書"
print "the query keyword:", keyword
for result in model.most_similar(keyword,topn=10):
            print result[0], result[1]

