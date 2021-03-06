# https://github.com/RaRe-Technologies/gensim/issues/1284
# -*- coding: utf-8 -*-

import os
import sys
import MeCab
# from numpy._distributor_init import NUMPY_MKL
import collections
import logging
import codecs
from gensim import models
from datetime import datetime
from gensim.models.doc2vec import LabeledSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


INPUT_DOC_DIR = r'C:\test\doc2vec\before'
OUTPUT_MODEL = r'C:\test\doc2vec\doc2vec_s400m1i10.model'
stopworddir = r'C:\test\doc2vec\result\mecab_stopword.txt'
PASSING_PRECISION = 93

ofile = open(stopworddir,"r")
ja_stopword = ofile.read().split()
ofile.close()

# 全てのファイルのリストを取得
def get_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

# ファイルから文章を返す
def read_document(path):
    # print path
    # with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:  # , encoding='sjis' , errors='ignore'
    with open(path, "r") as f:
        # print f.read()
        return f.read()

# 文章から単語に分解して返す
def split_into_words(doc, name=''):
    for word in ja_stopword:
        # print word
        try:
            doc = doc.replace(word, "").replace("\t", "").replace("\n", "")  # .replace("	","")
        except IndexError, e:
            print "error!", e
            print file
            continue
    mecab = MeCab.Tagger("-Ochasen")
    lines = mecab.parse(doc).splitlines()   #.encode('utf-8')
    words = []
    for line in lines:
        chunks = line.split('\t')
        if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):
            words.append(chunks[0])
    # print "\n"
    # print  "1", words
    return LabeledSentence(words=words, tags=[name])

# ファイルから単語のリストを取得
def corpus_to_sentences(corpus):
    docs = [read_document(x) for x in corpus]   #.decode('sjis')
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        sys.stdout.write('\r前処理中 {} / {}'.format(idx, len(corpus)))
        name = name.split("\\")[-1]  #.decode("shift_jisx0213")
        yield split_into_words(doc, name)

# 学習
def train(sentences):
    print('\n訓練開始')
    model = models.Doc2Vec(sentences, size=50, min_count=1, iter=10, workers=4)  #(sentences, size=50, min_count=1, iter=10)　　(sentences, size=50, alpha=0.0015, sample=1e-4, min_count=1, workers=4)  #   #, size=50, alpha=0.0015, sample=1e-4, min_count=2, workers=2)

    # print "len(sentences):", len(sentences)
    # for epoch in range(20):
    #     print('Epoch: {}'.format(epoch + 1))
    #     model.train(sentences, total_examples=len(sentences), epochs=5) #, total_examples=corpus_count, epochs=iter)
    #     model.alpha -= (0.025 - 0.0001) / 19
    #     model.min_alpha = model.alpha

    # model.build_vocab(sentences)
    # for x in range(30):
    #     print x
    #     model.train(sentences, total_examples=len(sentences), epochs=5)
    #     model.alpha -= 0.002  # decrease the learning rate`
    #     model.min_alpha = model.alpha  # fix the learning rate, no decay
    #     ranks = []
    #     for doc_id in range(100):
    #         inferred_vector = model.infer_vector(sentences[doc_id].words)
    #         sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    #         rank = [docid for docid, sim in sims].index(sentences[doc_id].tags[0])
    #         ranks.append(rank)
    #     print(collections.Counter(ranks))
    #     if collections.Counter(ranks)[0] >= PASSING_PRECISION:
    #         break
    return model

if __name__ == '__main__':
    begin = datetime.now()
    print begin
    corpus = list(get_all_files(INPUT_DOC_DIR))
    # for l in corpus:
    #     print l.decode('sjis')
    # print corpus
    sentences = list(corpus_to_sentences(corpus))
    print()
    model = train(sentences)
    model.save(OUTPUT_MODEL)
    end = datetime.now()
    print "Total procesing time: ",  end - begin
