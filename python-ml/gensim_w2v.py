#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:21:27 2018

@author: meicanhua
"""

# coding: utf-8
from gensim.models import word2vec
from gensim.models import Word2Vec
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
     if (len(sys.argv) != 3):
         print("Usage: python word_vector_train.py words_file model_file")
         sys.exit(1)
     
     sentences = word2vec.LineSentence(sys.argv[1])
     model = word2vec.Word2Vec(sentences,size=200,window=6,min_count=200,workers=32)
     #model = Word2Vec.load("../wikipedia_resume_sogounews_weixin_model_100/wikipedia_resume_sogounews_weixin_model_100")
     # model.train(sentences, total_examples=model.corpus_count,epochs=model.iter)
     model.save(sys.argv[2]+".model")
     wv = model.wv
     wv.save_word2vec_format(sys.argv[2]+".vector.bin",sys.argv[2]+".vocab.bin",binary=True)
     wv.save_word2vec_format(sys.argv[2]+".vector.txt",sys.argv[2]+".vocab.txt",binary=False)