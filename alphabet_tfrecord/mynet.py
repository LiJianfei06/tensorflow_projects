# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:51:42 2017

@author: root
"""


from gen_captcha import gen_captcha_text_and_image  
from gen_captcha import number  
from gen_captcha import alphabet  
from gen_captcha import ALPHABET  
import os
import sys
#import argparse # argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。
import random
from os import listdir
import os.path
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter,ImageEnhance
import time
from datetime import datetime
import numpy as np  
import tensorflow as tf  

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('train_size', 100000,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('test_size', 10000,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('Initial_learning_rate', 1e-2*128/256,
                            """max_steps.""")
tf.app.flags.DEFINE_integer('max_steps', 500000,
                            """max_steps.""")
tf.app.flags.DEFINE_string('log_dir', 'logs',
                           """log_dir.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """train_dir.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")





char_set = number + alphabet + ALPHABET  # 验证码长度  62
CHAR_SET_LEN = len(char_set) 

name_list =list(os.path.join(FLAGS.train_dir,name)for name in os.listdir(FLAGS.train_dir))
random_name_list=list(random.choice(name_list)for _ in range(1))
for root_str in random_name_list:
    text=(root_str.split('/')[-1]).split('.')[0][0]
    text = ''.join(text)
    image=np.array(Image.open(root_str)) 

MAX_CAPTCHA = len(text)
# 图像大小  
IMAGE_WIDTH = image.shape[0]  
IMAGE_HEIGHT = image.shape[1]  

print u"图像channel:",image.shape
print u"输出个数:", MAX_CAPTCHA   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐  




# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）  
def convert2gray(img):  
    if len(img.shape) > 2:  
        gray = np.mean(img, -1)  
        # 上面的转法较快，正规转法如下  
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]  
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  
        return gray  
    else:  
        return img  
    

# 文本转向量  
def text2vec(text):  
   
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)  #1*62
    
    for i in range(len(text)):
        if(ord(text[i])<=57 and ord(text[i])>=48):
            idx = i * CHAR_SET_LEN + ord(text[i])-48
        elif(ord(text[i])<=90 and ord(text[i])>=65):            # A....
            idx = i * CHAR_SET_LEN + ord(text[i])-65+10
        elif(ord(text[i])<=122 and ord(text[i])>=97):           # A....
            idx = i * CHAR_SET_LEN + ord(text[i])-97+36
        vector[idx] = 1  
    #print vector
    return vector  

# 向量转回文本  
def vec2text(vec):  
    text=[]  
    for c in vec:  
        #print c
        #char_at_pos = i #c/63  
        char_idx = c % CHAR_SET_LEN  
        if char_idx < 10:  
            char_code = char_idx + ord('0')  
        elif char_idx <36:  
            char_code = char_idx - 10 + ord('A')  
        elif char_idx < 62:  
            char_code = char_idx-  36 + ord('a')  
        else:  
            raise ValueError('error')  
        text.append(chr(char_code))  
    return "".join(text)  




#by LiJianfei
def get_next_batch(batch_size=128,root_str="train"):  
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])  
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])  
    i=0
    name_list =list(os.path.join(root_str,name)for name in os.listdir(root_str))
    random_name_list=list(random.choice(name_list)for _ in range(batch_size))
    for root_str in random_name_list:
        text=(root_str.split('/')[-1]).split('.')[0][0]
        text = ''.join(text)
        image=np.array(Image.open(root_str))  
        #print text,image
        batch_x[i,:] = 1.0-(image.flatten() *0.00390625) # (image.flatten()-128)/128  mean为0 
        batch_y[i,:] = text2vec(text)   
        if (((i+1)%(batch_size/100)==0) and (batch_size==FLAGS.train_size)):
            sys.stdout.write('\r>> loading samples... %.3f%%' % ((float(i+1) / float(batch_size)) * 100.0))
            sys.stdout.flush()                                                    # 这句代码的意思是刷新输出
        #print batch_x[i,:],batch_y[i,:]
        i=i+1
    return batch_x, batch_y 

####################################################################  
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/gpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)    #变量初始化
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))          # 截取的正态分布
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')  # L2 正则化
    tf.add_to_collection('losses', weight_decay)                            # 把变量放入一个集合，把很多变量变成一个列表
  return var


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    #这个函数产生正太分布，均值和标准差自己设定。默认:shape,mean=0.0,stddev=1.0,dtype=dtypes.float32,seed=None,name=None
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义CNN  
def inference(images,keep_prob):  
    x = tf.reshape(images, shape=[-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])  
   
    # conv layer  
    with tf.variable_scope('conv1') as scope:#每一层都创建于一个唯一的 tf.name_scope 之下,创建于该作用域之下的所有元素都将带有其前缀
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 1, 48],stddev=0.1,wd=0.00)  # 权值
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')                       # 实现卷积    
        biases = _variable_on_cpu('biases', [48], tf.constant_initializer(0.0))                 # 偏置
        conv1 = tf.nn.bias_add(conv, biases)           # 这个函数的作用是将偏差项 bias 加到 conv 上面
        print ('conv1',conv1)  
        #_activation_summary(conv1)
 
    # relu1
    relu1 = tf.nn.relu(conv1, name=scope.name)     # 激活函数      
    # pool1
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')          #局部响应归一化函数     
    tf.summary.histogram('norm1', norm1)
    print ('norm1',norm1)        #16*16
    
    
  # =============================================
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 48, 64],stddev=0.1,wd=0.00)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        conv2 = tf.nn.bias_add(conv, biases)
        print ('conv2',conv2)
        #_activation_summary(conv2)
    relu2 = tf.nn.relu(conv2, name=scope.name)
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')        
    norm2 = tf.nn.lrn(pool2, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    tf.summary.histogram('norm2', norm2)
    print ('norm2',norm2)       #8*8

  # ======================================
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 64, 64],stddev=0.1,wd=0.0)
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        conv3 = tf.nn.bias_add(conv, biases)
        print ('conv3',conv3)
        #_activation_summary(conv2)
    relu3 = tf.nn.relu(conv3, name=scope.name)
    pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool3')        
    norm3 = tf.nn.lrn(pool3, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    #pool3 = tf.nn.dropout(pool3, keep_prob)
    tf.summary.histogram('norm3', norm3)
    print ('norm3',norm3)       #4*4
    
   # ======================================
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 64, 64],stddev=0.1,wd=0.0)
        conv = tf.nn.conv2d(norm3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        conv4 = tf.nn.bias_add(conv, biases)
        print ('conv4',conv4)
        #_activation_summary(conv2)
    relu4 = tf.nn.relu(conv4, name=scope.name)
    pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool3')        
    #pool3 = tf.nn.dropout(pool3, keep_prob)
    norm4 = tf.nn.lrn(pool4, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    tf.summary.histogram('norm4', norm4)
    print ('norm4',norm4)       #2*2
    
   # ======================================
    with tf.variable_scope('fc101') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(norm4, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        print ('reshape',reshape)
        print ('dim',dim)
        weights = _variable_with_weight_decay('weights', shape=[2*2*64, 1024],stddev=1/1024.0, wd=0.00001)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.0))
        reshape = tf.reshape(pool4, [-1, weights.get_shape().as_list()[0]]) 
        relu101 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        drop101 = tf.nn.dropout(relu101, keep_prob)
        tf.summary.histogram('relu101', relu101)
        print ('relu101',relu101)
        #_activation_summary(local3)  
        
   # ======================================    
#    with tf.variable_scope('fc102') as scope:
#        weights = _variable_with_weight_decay('weights', shape=[256, 256],stddev=1/256.0, wd=0.0000)
#        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
#        relu102 = tf.nn.relu(tf.matmul(drop101, weights) + biases, name=scope.name)
#        drop102 = tf.nn.dropout(relu102, keep_prob)    
#        print ('relu102',relu102)

        
   # ======================================
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [1024, MAX_CAPTCHA*CHAR_SET_LEN],stddev=1/1024.0, wd=0.0)
        biases = _variable_on_cpu('biases', [MAX_CAPTCHA*CHAR_SET_LEN],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(drop101, weights), biases, name=scope.name)
        tf.summary.histogram('softmax_linear', softmax_linear)
        #_activation_summary(softmax_linear)

    #out = tf.nn.softmax(out)  
    return softmax_linear  

def loss_fun(logits, labels):
    #labels = tf.to_int64(labels)
    cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    #return cross_entropy_mean   #实现一个列表的元素的相加
    return tf.add_n(tf.get_collection('losses'), name='total_loss')   #实现一个列表的元素的相加


def evaluation(logits, labels):
    max_idx_p = tf.argmax(tf.reshape(logits, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
    max_idx_l = tf.argmax(tf.reshape(labels,      [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
    eval_correct=tf.equal(max_idx_p, max_idx_l)  
    return tf.reduce_mean(tf.cast(eval_correct, tf.float32)) 

def training(loss,global_step):
       
      # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.Initial_learning_rate,        #0.01
                                  global_step,                  #0
                                  int(FLAGS.train_size/FLAGS.batch_size),                  #136500
                                  0.96,   #0.1
                                  staircase=False)
    tf.summary.scalar('learning_rate', lr)


    train_op = tf.train.AdamOptimizer(lr).minimize(loss)  
#    print "train_op:",train_op
    return train_op,lr
