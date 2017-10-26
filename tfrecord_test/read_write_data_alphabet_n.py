# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf 
from PIL import Image
import numpy as np 
cwd = os.getcwd()     # get dir of this file

'''

...
'''
MAX_CAPTCHA = 4
CHAR_SET_LEN=62*MAX_CAPTCHA


i=0
classes = ["read_test_alphabet4" ];#目录 分别改成train 和 val 各跑一次就行了
print "start..."
writer = tf.python_io.TFRecordWriter("read_test_alphabet4.tfrecords")   #分别改成train 和 val 各跑一次就行了
for index, name in enumerate(classes):
    print index, name
    print cwd
    class_path = cwd + "/"+name + "/"
    print class_path
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        #print img_name
   
        img = Image.open(img_path)
        #img = img.resize((32, 32))
        img_raw = img.tobytes()              #将图片转化为原生bytes
        i+=1
        label_val=0
        for n in range(MAX_CAPTCHA):
            vector = np.zeros(1*CHAR_SET_LEN,int)  #4*62
            if(ord(img_name[n])<=57 and ord(img_name[n])>=48):
                idx = ord(img_name[n])-48
            elif(ord(img_name[n])<=90 and ord(img_name[n])>=65):            # A....
                idx = ord(img_name[n])-65+10
            elif(ord(img_name[n])<=122 and ord(img_name[n])>=97):           # a....
                idx = ord(img_name[n])-97+36
            idx+=10
            label_val+=idx*pow(100, n)
            #print label_val
#        vector[idx] = 1  
        #print vector.tolist()
        #idx=i
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_val])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()



for serialized_example in tf.python_io.tf_record_iterator("read_test_alphabet4.tfrecords"):   #分别改成train 和 val 各跑一次就行了
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    # 可以做一些预处理之类的
    #print image, label
    
  
    
    
def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 32, 1])                      # check!
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    print "OK!"
    return img, label



#分别改成train 和 val 各跑一次就行了
img, label = read_and_decode("read_test_alphabet4.tfrecords")   

#使用shuffle_batch可以随机打乱输入
img_batch_shuffle, label_batch_shuffle = tf.train.shuffle_batch([img, label],
                                                batch_size=10, capacity=200,
                                                min_after_dequeue=100,num_threads=1)
img_batch, label_batch = tf.train.batch([img, label],
                                                batch_size=10, capacity=200,
                                                num_threads=1)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(10):
        val, l= sess.run([img_batch_shuffle, label_batch_shuffle])   # 打乱
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12) 
        print(val.shape, l)
        #print val[0,:,:,0]
