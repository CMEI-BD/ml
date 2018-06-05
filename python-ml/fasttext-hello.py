#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:15:23 2018

@author: meicanhua
"""

import fasttext
import jieba
import os
import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
#from urllib.request import urlopen

from cStringIO import StringIO



reload(sys)
sys.setdefaultencoding('utf-8')
y = ['筛选通过','筛选失败']
basedir = "/Users/didi/workspace/data/tensorflow/resume_data"

def readPDF(pdfFile):
    #创建一个PDF资源管理器对象来存储共享资源
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    #设定参数进行分析
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec = 'utf-8', laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    fp = open(pdfFile, 'rb')
    #StringIO(pdfFile)
    for page in PDFPage.get_pages(fp, set(), maxpages=0, password="", caching=True, check_extractable=True):
        interpreter.process_page(page) 
    #process_pdf(rsrcmg, device, pdfFile) 2011版过期函数
    fp.close()
    device.close()
    content = retstr.getvalue()
    retstr.close()
    return content
    

    

def convertFastTextFileFormat(inDir, outFilePath, yclass):
    #print(inDir)
    inFiles = os.listdir(inDir)
    outFile = open(outFilePath, "a+")
    for infile in inFiles:
        path = inDir+"/"+infile
        print(path)
        if (infile.endswith(".pdf")): 
            #with open(path, 'r') as fr:
            text = readPDF(path)
            #fr.read()
            #print text
            text = text.decode("utf-8").encode("utf-8")
            segText = jieba.cut(text.replace("\t", " ").replace("\n", " "))
            outline = " ".join(segText)
            outline = outline.encode("utf-8") + "\t__label__" + yclass +"\n"
            outFile.write(outline)
            outFile.flush()
    outFile.close()
    
    
def trainWord2Vector():
    model1 = fasttext.skipgram('/Users/didi/workspace/data/tensorflow/resume_data/resume_train.txt', 'model', lr=0.01, dim=300)
    #for word in model1.words:
    print model1.words['模型'].encode('utf-8')

#    model2 = fasttext.cbow('/Users/didi/workspace/data/tensorflow/resume_data/resume_train.txt', 'model')
#    print model2.words  

def trainSupervised():
    classifier1 = fasttext.supervised('/Users/didi/workspace/data/tensorflow/resume_data/resume_train.txt',
                                     '/Users/didi/workspace/data/tensorflow/model/resume.model', label_prefix='__label__') 
    
    classifier2 = fasttext.load_model('/Users/didi/workspace/data/tensorflow/model/resume.model.bin', label_prefix='__label__')
    result =classifier2.test('/Users/didi/workspace/data/tensorflow/resume_data/resume_train.txt')
    print(result.precision)
    print(result.recall)

if __name__ == '__main__':
    #pdfFile = urlopen("http://pythonscraping.com/pages/warandpeace/chapter1.pdf")
    #convertFastTextFileFormat("/Users/didi/Desktop/实习生/fail", "/Users/didi/workspace/data/tensorflow/resume_data/resume_train.txt", y[1])  
    trainWord2Vector()
    #trainSupervised()
    
    
    
        

    
    
    


