# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf 
from PIL import Image
import numpy as np 
cwd = os.getcwd()     # get dir of this file

'''

 ------------------------------
测试字母（n位）
...
'''
 
    





def save_tfrecords(train_or_test):
    i=0
    classes = [train_or_test];#目录 分别改成train 和 val 各跑一次就行了
    writer = tf.python_io.TFRecordWriter(train_or_test+".tfrecords")   #分别改成train 和 val 各跑一次就行了
    for index, name in enumerate(classes):
        #print index, name
        #print cwd
        #class_path = cwd + "/"+name + "/"
        class_path = "/home/lijianfei/datasets/cifar10_40/"+train_or_test+"/"
        print class_path
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            #print img_name[0],img_name[1]
       
            img = Image.open(img_path)
            #img = img.resize((32, 32))
            img_raw = img.tobytes()              #将图片转化为原生bytes
            i+=1
            label_val=0

            if img_name[0:4]=="airp":
                idx=0
            elif img_name[0:4]=="auto":
                idx=1
            elif img_name[0:4]=="bird":
                idx=2
            elif img_name[0:3]=="cat":
                idx=3
            elif img_name[0:4]=="deer":
                idx=4
            elif img_name[0:3]=="dog":
                idx=5
            elif img_name[0:4]=="frog":
                idx=6
            elif img_name[0:4]=="hors":
                idx=7
            elif img_name[0:4]=="ship":
                idx=8
            elif img_name[0:4]=="truc":
                idx=9

            label_val=idx
           

            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_val])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()


    for serialized_example in tf.python_io.tf_record_iterator(train_or_test+".tfrecords"):   #分别改成train 和 val 各跑一次就行了
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        image = example.features.feature['image'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的
        #print image, label
        
 




    
def read_and_decode(filename,w_h):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),})

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [w_h, w_h, 3])                      # check!
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    #print "OK!"
    return img, label








if __name__ == '__main__':
    print "make train datasets..."
    save_tfrecords('train')
    print "seccuss!"

    print "make test datasets..."
    save_tfrecords('test')
    print "seccuss!"

    #分别改成train 和 val 各跑一次就行了
    img_train, label_train = read_and_decode("train.tfrecords",40)   
    img_test, label_test = read_and_decode("test.tfrecords",32)   


    #img_batch, label_batch = tf.train.batch([img_train, label_train],
    #                                                batch_size=10, capacity=200,
    #                                                num_threads=1)

    #使用shuffle_batch可以随机打乱输入
    img_batch_shuffle_train, label_batch_shuffle_train = tf.train.shuffle_batch([img_train, label_train],
                                                    batch_size=10, capacity=200,
                                                    min_after_dequeue=100,num_threads=4)
    img_batch_shuffle_test, label_batch_shuffle_test = tf.train.shuffle_batch([img_test, label_test],
                                                    batch_size=10, capacity=200,
                                                    min_after_dequeue=100,num_threads=4)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(10):
            #val, l= sess.run([img_batch, label_batch])   # 打乱
            val_train, l_train= sess.run([img_batch_shuffle_train, label_batch_shuffle_train])   # 打乱
            val_test, l_test= sess.run([img_batch_shuffle_test, label_batch_shuffle_test])   # 打乱
            #我们也可以根据需要对val， l进行处理
            #l = to_categorical(l, 12) 
            print(val_train.shape, l_train)
            print(val_test.shape, l_test)
            #print val[0,:,:,0]








