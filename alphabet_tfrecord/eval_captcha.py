# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 08:20:54 2017

@author: root
"""

from gen_captcha import gen_captcha_text_and_image  
from gen_captcha import number  
from gen_captcha import alphabet  
from gen_captcha import ALPHABET  
import os
import sys
import argparse # argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。
from os import listdir
import os.path
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter,ImageEnhance
import time
import random
from datetime import datetime
import numpy as np  
import tensorflow as tf  
import mynet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('test_size_eval', 1000,
                            """Number of images to test.""")

print "start..."
def crack_captcha():  
    with tf.Graph().as_default():
        
        with tf.device('/cpu:0'):
            X = tf.placeholder(tf.float32, [None, mynet.IMAGE_HEIGHT*mynet.IMAGE_WIDTH])  
            #Y = tf.placeholder(tf.float32, [None, mynet.MAX_CAPTCHA*mynet.CHAR_SET_LEN])
            
        keep_prob = tf.placeholder(tf.float32) # dropout 
        
        logits = mynet.inference(X,keep_prob)  
#   
        saver = tf.train.Saver()  
        with tf.Session() as sess:  
            saver.restore(sess,"./logs/model.ckpt-110000")  
       
            predict = tf.argmax(tf.reshape(logits, [-1, mynet.MAX_CAPTCHA, mynet.CHAR_SET_LEN]), 2)  
            
            my_root="test"
            name_list =list(os.path.join(my_root,name)for name in os.listdir(my_root))
            random_name_list=list(random.choice(name_list)for _ in range(FLAGS.test_size_eval))
            
            ture_mun=0.0
            false_mun=0.0
            for root_str in random_name_list:
                text=(root_str.split('/')[-1]).split('.')[0][0:4]
                text = ''.join(text)
                image=np.array(Image.open(root_str)) 
                #image = convert2gray(image) 
                image = (image.flatten() *0.00390625)
                print text
                
                text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})  
                predict_text = text_list[0].tolist()  
                
                text_pre=mynet.vec2text(predict_text)
                if (((abs(ord(text[0])-ord(text_pre[0]))==32 and ord(text_pre[0])>48) or (ord(text[0])-ord(text_pre[0])==0))):
                    hint="True" 
                    ture_mun+=1
                else:
                    hint="False"
                    false_mun+=1        
                print("正确: {}  预测: {} check:{}".format(text[0], text_pre,hint))
            print("accuracy:%.4f"%(ture_mun/(ture_mun+false_mun)))
         
   


def main(argv=None):  # pylint: disable=unused-argument
  crack_captcha()                                            


if __name__ == '__main__':
  tf.app.run()
