# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:32:34 2017
设置好各自数量，直接执行即可
@author: root
"""

from gen_captcha import gen_captcha_text_and_image  
from gen_captcha import number  
from gen_captcha import alphabet  
from gen_captcha import ALPHABET  
from PIL import Image 
import shutil
from numpy import *
import os
import sys
from os import listdir
import os.path
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter,ImageEnhance
import time

from datetime import datetime
import random

train_mun = 100000 
val_mun = 10000 
test_mun = 1000 




#generate
def random_captcha_generate(num=2,file_train_flag=1,file_val_flag=0,file_test_flag=0):
    for i in range(num):
        text, image = gen_captcha_text_and_image(file_train_flag,file_val_flag,file_test_flag,i=i)
        #print i,text
        if (i+1)%(num/100)==0:
            if file_train_flag==1:
                sys.stdout.write('\r>> Generate train samples... %.3f%%' % (float(i+1) / float(num) * 100.0)) 
            elif file_val_flag==1:
                sys.stdout.write('\r>> Generate val samples... %.3f%%' % (float(i+1) / float(num) * 100.0)) 
            elif file_test_flag==1:
                sys.stdout.write('\r>> Generate test samples... %.3f%%' % (float(i+1) / float(num) * 100.0))             
            sys.stdout.flush()                                                    # 这句代码的意思是刷新输出
    print "\n %d samples generate succeed!"%num   
    
 

#resize
def random_captcha_resize(str_place):
    #filename1="val.txt"
    
    #fp=open(filename1,"w")
    i=0
    for dirpath, dirnames, filenames in os.walk(str_place):
        print "Directory:%s"%dirpath
        #print type(filenames)    #返回的是一个list
        file_mun=len(filenames)
        for filename in filenames:
            if (i+1)%(file_mun/100)==0:
                sys.stdout.write('\r>> resize Directory samples... %.3f%%' % (float(i+1) / float(file_mun) * 100.0)) 
                sys.stdout.flush()                                                    # 这句代码的意思是刷新输出
            #print i,filename,filename[0]
            i=i+1
            img = Image.open(str_place+filename)
            img = img.crop((0,0,40,60))  #  裁剪
            img=img.convert('L')     #灰度化
            #img=img.convert('1')#二值化
#            for ii in range(2):
#                img=img.filter(ImageFilter.MedianFilter) #中值滤波
            
            out = img.resize((32,32),Image.ANTIALIAS)
            out.save(str_place+filename,quality=100)
    
    #        if(ord(filename[0])<=57 and ord(filename[0])>=48):
    #            fp.write(filename+' '+filename[0]+'\n')
    #        elif(ord(filename[0])<=90 and ord(filename[0])>=65):            # A....
    #            fp.write(filename+' '+str(ord(filename[0])-65+10)+'\n')
    #        elif(ord(filename[0])<=122 and ord(filename[0])>=97):           # A....
    #            fp.write(filename+' '+str(ord(filename[0])-97+36)+'\n')     
    #fp.close()
    print "\n succeed!"



start_time = time.time()

#if os.path.exists('train'):                # 如果目录存在
#    shutil.rmtree(r'train') 
#if os.path.exists('val'):                # 如果目录存在
#    shutil.rmtree(r'val') 
if os.path.exists('test'):                # 如果目录存在
    shutil.rmtree(r'test') 

#os.makedirs("train")
#os.makedirs("val")
os.makedirs("test")
#random_captcha_generate(num=train_mun,file_train_flag=1,file_val_flag=0,file_test_flag=0)
#random_captcha_generate(num=val_mun,file_train_flag=0,file_val_flag=1,file_test_flag=0)
random_captcha_generate(num=test_mun,file_train_flag=0,file_val_flag=0,file_test_flag=1)


#random_captcha_resize("train/")
#random_captcha_resize("val/")
random_captcha_resize("test/")

duration = time.time() - start_time

print('Spend time: %.3f sec' % (duration))























