# -*- coding: utf-8 -*-


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
from datetime import datetime
import numpy as np  
import tensorflow as tf  
import random
#reload(mynet)
import mynet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('test_batch', 100,
                            """test_batch*test_num should equal to number of test size.""")
tf.app.flags.DEFINE_float('test_num', 100,
                            """test_batch*test_num should equal to number of test size....""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """log_frequency.""")
tf.app.flags.DEFINE_string('test_dir', 'val',
                           """test_dir.""")

clear=False   # 是否清空从头训练


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
    img = tf.reshape(img, [32, 64, 1])        # check!
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    img = 1-tf.cast(img, tf.float32) * 0.00390625

    label = tf.cast(features['label'], tf.int32)
    print "OK!"
    return img, label




# 训练  
def train_crack_captcha_cnn():  
#    batch_x_train, batch_y_train = mynet.get_next_batch(batch_size=FLAGS.train_size,root_str=FLAGS.train_dir) 
#    batch_x_test, batch_y_test = mynet.get_next_batch(batch_size=FLAGS.test_size,root_str=FLAGS.test_dir) 
    start_time = time.time()
    total_duration_time=0      # total_time

        
    with tf.Graph().as_default():    
        with tf.device('/gpu:0'):
            X = tf.placeholder(tf.float32, [None, mynet.IMAGE_HEIGHT*mynet.IMAGE_WIDTH])  
            Y = tf.placeholder(tf.float32, [None, mynet.MAX_CAPTCHA*mynet.CHAR_SET_LEN])
            
        batch_x_train, batch_y_train = read_and_decode("train.tfrecords")
        batch_x_test, batch_y_test = read_and_decode("val.tfrecords")
        
        This_batch = tf.placeholder(tf.int32) # dropout
        print "batch_x_train:",batch_x_train
        print "batch_y_train:",batch_y_train

        img_batch_train, label_batch_train = tf.train.shuffle_batch([batch_x_train, batch_y_train],
                                            batch_size=This_batch, capacity=1024,
                                            min_after_dequeue=128,num_threads=16) 
        img_batch_test, label_batch_test = tf.train.batch([batch_x_test, batch_y_test],
                                            batch_size=This_batch, capacity=2048,
                                            #min_after_dequeue=512,
                                            num_threads=16) 
            
        global_ = tf.Variable(tf.constant(0))  
        keep_prob = tf.placeholder(tf.float32) # dropout 
        
        logits = mynet.inference(X,keep_prob)  
        #print 222
        loss=mynet.loss_fun(logits=logits, labels=Y)
        tf.summary.scalar('loss', loss)
        train_op,lr = mynet.training(loss,global_step=global_)
        
        accuracy = mynet.evaluation(logits=logits, labels=Y)
        
        tf.summary.scalar('accurate', accuracy) # display accurate in TensorBoard
        
        summary = tf.summary.merge_all()
        
        saver = tf.train.Saver() 
        
        
        
        #sess = tf.Session()
#------------------------------------------- 按比例限制显存       
#        config = tf.ConfigProto()
#        config.gpu_options.per_process_gpu_memory_fraction = 0.4
#        sess = tf.Session(config=config)
#-------------------------------------------        
 #------------------------------------------- 按需求增长显存       
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config,)
#-------------------------------------------    
        
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer_train = tf.summary.FileWriter(FLAGS.log_dir+"/train", sess.graph)
        summary_writer_test = tf.summary.FileWriter(FLAGS.log_dir+"/test", sess.graph)
        
        #with tf.Session() as sess:  
        init_op = tf.global_variables_initializer()        
        if clear==True:
#------------------------------------------- 从头训练     
            sess.run(init_op)     
            init_step=0
        else:
#------------------------------------------- 断电继续(利用之前训练好的sess继续训练)    
            saver.restore(sess, "logs/model.ckpt-200000")  #要改
            print "Model restored."  
            init_step=200000                               #要改
            
            
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        
        duration = time.time() - start_time
        total_duration_time+=duration
        start_time = time.time()
        acc=0
        for step in range(init_step,FLAGS.max_steps):
            #batch_x, batch_y = mynet.get_next_batch(batch_size=FLAGS.batch_size,root_str=FLAGS.train_dir) 
            
#            k_index=np.array(random.sample(range(FLAGS.train_size),FLAGS.batch_size))
#            batch_x=batch_x_train[k_index]
#            batch_y=batch_y_train[k_index]
            #print "\n\n\n\n"

            batch_x, l= sess.run([img_batch_train, label_batch_train], feed_dict={This_batch: FLAGS.batch_size})
            batch_y= np.zeros([FLAGS.batch_size,mynet.MAX_CAPTCHA*mynet.CHAR_SET_LEN])  #1*62
            for i in range(FLAGS.batch_size):
                for j in range(mynet.MAX_CAPTCHA):
                    batch_y[i,(l[i])%pow(100,j+1)/pow(100,j)-10+mynet.CHAR_SET_LEN*j] = 1 
            batch_x = np.reshape(batch_x, (-1, mynet.IMAGE_HEIGHT*mynet.IMAGE_WIDTH)) 
            
            #print type(batch_x),type(batch_y)
            #print batch_x.shape,batch_y.shape
            #sys.exit(0)
            _, learn_lr,loss_ = sess.run([train_op,lr, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5+float(step)/2.0/float(FLAGS.max_steps),global_:step})  
            #print(step, loss_)  
            if (step+1) % FLAGS.log_frequency == 0:
                duration = time.time() - start_time
                total_duration_time+=duration
                start_time = time.time()
                
                print "ljf-tf:[%s] Iteration:%d/%d (%.1f examples/sec, %.3f sec/%.3f iters) ,loss=%.5f ,lr=%.5f"%(datetime.now(),(step+1),FLAGS.max_steps,FLAGS.batch_size*FLAGS.log_frequency/duration,duration,FLAGS.log_frequency,loss_,learn_lr)


                           
#                k_index=np.array(random.sample(range(FLAGS.test_size),FLAGS.test_batch))
#                batch_x=batch_x_test[k_index]
#                batch_y=batch_y_test[k_index]         
#                batch_x, l= sess.run([img_batch_train, label_batch_train], feed_dict={This_batch: FLAGS.test_batch})
#                batch_y= np.zeros([FLAGS.test_batch,mynet.MAX_CAPTCHA*mynet.CHAR_SET_LEN])  #1*62
#                for i in range(FLAGS.test_batch):
#                    for j in range(mynet.MAX_CAPTCHA):
#                        batch_y[i,(l[i])%pow(100,j+1)/pow(100,j)-10+mynet.CHAR_SET_LEN*j] = 1         
#                batch_x = np.reshape(batch_x, (-1, mynet.IMAGE_HEIGHT*mynet.IMAGE_WIDTH)) 
                
                summary_str = sess.run(summary, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.,global_:step})
                summary_writer_train.add_summary(summary_str, step)
                summary_writer_train.flush()

#                k_index=np.array(random.sample(range(FLAGS.test_size),FLAGS.test_batch))
#                batch_x=batch_x_test[k_index]
#                batch_y=batch_y_test[k_index]   

            # 每100 step计算一次准确率  
                if (((step+1) % (FLAGS.log_frequency*5)== 0) or (step==0)): 
                    acc=0
                    for i in range(FLAGS.test_num):    #测试集 扫完
    #                    k_index=np.array(random.sample(range(i*FLAGS.test_batch,(i+1)*FLAGS.test_batch),FLAGS.test_batch))
    #                    batch_x=batch_x_train[k_index]
    #                    batch_y=batch_y_train[k_index]                        
                        batch_x, l= sess.run([img_batch_test, label_batch_test], feed_dict={This_batch: FLAGS.test_batch})
                        batch_y= np.zeros([FLAGS.test_batch,mynet.MAX_CAPTCHA*mynet.CHAR_SET_LEN])  #1*62
                        for i in range(FLAGS.test_batch):
                            for j in range(mynet.MAX_CAPTCHA):
                                batch_y[i,(l[i])%pow(100,j+1)/pow(100,j)-10+mynet.CHAR_SET_LEN*j] = 1           
                        batch_x = np.reshape(batch_x, (-1, mynet.IMAGE_HEIGHT*mynet.IMAGE_WIDTH)) 
                      
                        acc += sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.,global_:step})   
                    acc=acc/FLAGS.test_num
                    
                    duration = time.time() - start_time
                    total_duration_time+=duration
                    start_time = time.time()
                    print "TensorFlow test net output: test accuracy:%.4f%% %.3f sec/1 test keep_prob:%.3f  test time:%.3f sec  total time:%.3f sec)"%(acc*100,duration,0.5+float(step)/2.0/float(FLAGS.max_steps),duration,total_duration_time)       
                    
                else:
                    batch_x, l= sess.run([img_batch_test, label_batch_test], feed_dict={This_batch: FLAGS.test_batch})
                    batch_y= np.zeros([FLAGS.test_batch,mynet.MAX_CAPTCHA*mynet.CHAR_SET_LEN])  #1*62
                    for i in range(FLAGS.test_batch):
                        for j in range(mynet.MAX_CAPTCHA):
                            batch_y[i,(l[i])%pow(100,j+1)/pow(100,j)-10+mynet.CHAR_SET_LEN*j] = 1           
                    batch_x = np.reshape(batch_x, (-1, mynet.IMAGE_HEIGHT*mynet.IMAGE_WIDTH)) 
                
                summary_str = sess.run(summary, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.,global_:step})
                summary_writer_test.add_summary(summary_str, step)
                summary_writer_test.flush()                            





            if (step+1) % 10000 == 0: 
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step+1)  

        coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()           






def main(argv=None):  # pylint: disable=unused-argument
  if clear==True:
      if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
      tf.gfile.MakeDirs(FLAGS.log_dir)
  train_crack_captcha_cnn()                                            


if __name__ == '__main__':
  tf.app.run()
