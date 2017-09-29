# -*- coding: utf-8 -*-
from gensim import models
from datetime import datetime
from gensim.models.doc2vec import LabeledSentence
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

INPUT_DOC_DIR = r'C:\test\word2vec\text'
OUTPUT_MODEL = r'C:\test\doc2vec\doc2vec_s400m1i10.model'
model = models.Doc2Vec.load(OUTPUT_MODEL)

query = unicode('1.txt',"utf-8")

print "the query file name:", query
for i in model.docvecs.most_similar(query, topn=10):
    print i[0], i[1]

keyword = "番組"
print "the query keyword:", keyword
for result in model.most_similar(keyword,topn=10):
            print result[0], result[1]

