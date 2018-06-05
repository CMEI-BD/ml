#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:19:51 2018

@author: meicanhua
"""

import jieba
import jieba.posseg as pseg
import multiprocessing
import os
import time
import sys

# 多进程跑
# jieba.enable_parallel(multiprocessing.cpu_count())

stopwords_nature = ["m","mq","mg","b","begin","bg","bl","c","cc","e","end","o","p","pba","pbei","q","qt","qv","u",
					"ude1","ude2","ude3","udeng","udh","uguo","ule","ulian","uls","usuo","uyy","uzhe","uzhi","y","z",
					"r","rr","ry","rys","ryt","ryv","rz","rzs","rzt","rzv","w","nx"]

with open("stopwords.txt", "r") as f:
	stopwords = [x.strip() for x in f.readlines()]

def load_dict(dict_path):
	for dict in os.listdir(dict_path):
		jieba.load_userdict(dict_path + "/" + dict)

def engine(infile, outfile):
	line_number = 1
	with open(infile, 'r', errors='ignore') as f:
		with open(outfile, 'a') as g:
			for line in f:
				print("正在对第{0}行分词".format(str(line_number)))
				word_nature = pseg.cut(line.strip())
				for word, nature in word_nature:
					if word not in stopwords and nature not in stopwords_nature:
						g.write(word + " ")
				g.write("\n")
				line_number += 1

if __name__ == "__main__":
	#infile = input(">Enter infile path:")
	infile = sys.argv[1]
	outfile = "cut_" + infile
	load_dict("custom_dict")
	engine(infile, outfile)