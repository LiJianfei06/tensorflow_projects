# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 08:20:54 2017

@author: root
"""


import cv2
import os
import sys
#import argparse # argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。
from os import listdir
import os.path
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter,ImageEnhance
#import time
import random
from datetime import datetime
import numpy as np  
import tensorflow as tf  
import ResNet
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages 
slim = tf.contrib.slim # 使用方便的contrib.slim库来辅助创建ResNet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('test_size_eval', 10000,
                            """Number of images to test.""")

print "start..."
def crack_captcha(model_name):  
    with tf.Graph().as_default():
        _inputRGB = tf.placeholder(tf.float32,[None, ResNet.IMAGE_HEIGHT, ResNet.IMAGE_WIDTH,3 ])            
        keep_prob = tf.placeholder(tf.float32) # dropout 
        is_train = tf.placeholder(tf.bool) 
 
        with slim.arg_scope(ResNet.resnet_arg_scope(is_training=is_train)): # is_training设置为false
            logits, end_points = ResNet.resnet_20(_inputRGB, 10,is_training=is_train)
        
            #print "logits:",logits#,end_points   
            #print "end_points['predictions'] :",end_points['predictions'] #,end_points   
            #sys.exit(0)
        saver = tf.train.Saver()  
         #------------------------------------------- 按需求增长显存       
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config,)
        
        saver.restore(sess,model_name)          # 读取模型  
   
        #predict = tf.argmax(tf.reshape(logits, [-1, 1, 10]), 2)  slim.softmax(logits)
        predict1 = tf.argmax(tf.reshape(end_points['predictions'] , [-1, 1, 10]), 2)  
        
        #my_root="test/image"
        my_root="/home/lijianfei/datasets/cifar10_40/test/"
        #my_root="../cifar10_tfrecord_VggNet/val"
        name_list =list(os.path.join(my_root,name)for name in os.listdir(my_root))
        #random_name_list=list(random.choice(name_list)for _ in range(FLAGS.test_size_eval))
        
        
        labels_filename = './labels.txt'
        labels = np.loadtxt(labels_filename, str, delimiter='\t')
        
        ture_mun=0.0
        false_mun=0.0
        for root_str in name_list:
            text=(root_str.split('/')[-1]).split('.')[0][0:4]
            text = ''.join(text)
            #image=np.array(Image.open(root_str)) 
            #image = convert2gray(image) 
            bgrImg_temp = cv2.imread(root_str)#* (1.0 / 255.0) - 0.5
            #bgrImg_temp = bgrImg_temp * (1.0 / 255.0) 
            #bgrImg_temp=cv2.resize(bgrImg_temp,(28,28))
            rgbImg=cv2.cvtColor(bgrImg_temp, cv2.COLOR_BGR2RGB)
            #img = tf.cast(rgbImg, tf.float32) * (1. / 255)
            img = rgbImg[:][:][:]* (1. / 255.0)
            #print "img:",img
            #ROI = rgbImg[2:2+28, 2:2+28]  
            #image = (image.flatten() *0.00390625)-0.5
            #print text
            #print bgrImg_temp.shape
            
            text_list = sess.run(predict1, feed_dict={_inputRGB: [img],is_train:False, keep_prob: 1.0})  
 
            predict_text = text_list[0].tolist()  

            if labels[predict_text][0][0:3]== root_str.split("/")[-1][0:3]:
                hint="True" 
                ture_mun+=1.
            else:
                hint="False"
                false_mun+=1.  
            #if root_str.split("/")[-1][0:3]== "shi":  
            print("正确: %20s 预测: %10s /check:%5s"%(root_str.split("/")[-1].split(".")[0], labels[predict_text][0],hint))
        print("%d images  true:%d  false:%d   accuracy:%.4f%%"%(ture_mun+false_mun,ture_mun,false_mun,ture_mun*100.0/(ture_mun+false_mun)))
     
   


def main(argv=None):  # pylint: disable=unused-argument
  model_name="./logs/model.ckpt-64000"
  crack_captcha(model_name)                                            


if __name__ == '__main__':
  tf.app.run()
