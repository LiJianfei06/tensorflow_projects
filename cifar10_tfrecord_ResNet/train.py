# -*- coding: utf-8 -*-

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
import sklearn.preprocessing as prep    # 提供了各种公共函数
#reload(ResNet)
import collections # 原生的collections库
import tensorflow as tf
slim = tf.contrib.slim # 使用方便的contrib.slim库来辅助创建ResNet

import ResNet


def load_weights( weight_file, sess):
    parameters = []
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    for i, k in enumerate(keys):
       # if i not in [30,31]:
        sess.run(parameters[i].assign(weights[k]))
    print("-----------all done---------------")


# Basic model parameters.
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('train_size', 50000,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('test_size', 10000,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('test_batch', 20,
                            """test_batch*test_num should equal to number of test size.""")
tf.app.flags.DEFINE_float('test_num', 500,
                            """test_batch*test_num should equal to number of test size....""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_float('Initial_learning_rate', 0.1,
                            """Initial_learning_rate.""")
tf.app.flags.DEFINE_integer('max_steps', 64000,
                            """max_steps.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """log_frequency.""")
tf.app.flags.DEFINE_string('test_dir', 'val',
                           """test_dir.""")
tf.app.flags.DEFINE_string('log_dir', 'logs',
                           """log_dir.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
#tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")
                           """train_dir.""")

clear=True   # 是否清空从头训练
is_on_subdivisions = True
subdivisions = 1
subdivisions_batch_size = int(np.ceil(FLAGS.batch_size / subdivisions))








"""解析rfrecord """
def read_and_decode(filename,w_h,phase="train"):
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
    img = tf.reshape(img, [w_h, w_h,3])
    
    if phase=="train":
        img = tf.image.random_flip_left_right(img)
    #img = tf.image.flip_up_down(img)
        
        #img = tf.image.random_brightness(img, 0.2)          # 亮度
        #img = tf.image.random_contrast(img, 0.1,0.2)        # 对比度
        #img = tf.image.random_hue(img, 0.2)                 # 色相
        #img = tf.image.random_saturation(img, 0.1,0.2)      # 饱和度


    img = tf.cast(img, tf.float32) * (1. / 255)
    distorted_image = tf.random_crop(img,[ResNet.IMAGE_HEIGHT,ResNet.IMAGE_WIDTH,3])
    distorted_image = tf.cast(distorted_image, tf.float32)

    label = tf.cast(features['label'], tf.int32)
    print "OK!"
    return distorted_image, label



"""学习策略"""
"""return:学习率"""
def training(loss,global_step):
    boundaries = [32000, 48000, 64000]
    values = [FLAGS.Initial_learning_rate, FLAGS.Initial_learning_rate/10.0, FLAGS.Initial_learning_rate/100.0,FLAGS.Initial_learning_rate/100.0]
    lr = tf.train.piecewise_constant(global_step, boundaries, values)
    
    opt = tf.train.MomentumOptimizer(lr,momentum=0.9,use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([tf.group(*update_ops)]):
        #train_op = slim.learning.create_train_op(loss, opt, global_step)
        #train_op = optimizer.minimize(loss)   
        train_op = slim.learning.create_train_op(loss, opt,global_step=tf.train.get_or_create_global_step())
    return train_op,lr


"""学习策略"""
"""return:损失"""
def loss_fun(logits, labels):
  #logits1=tf.reshape(logits, [-1, 10])
  pose_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
  #slim.losses.add_loss(pose_loss)                  #will be removed after 2016-12-30.
  tf.losses.add_loss(pose_loss)
  #regularization_loss = tf.add_n(slim.losses.get_regularization_losses()) 
  #total_loss2 =  pose_loss + regularization_loss 
  #total_loss2 = slim.losses.get_total_loss()       #will be removed after 2016-12-30
  total_loss2 = tf.losses.get_total_loss()
  #cross_entropy_mean = tf.reduce_mean(total_loss2)  
  return total_loss2


"""return:top_k_精度(分类)"""
def top_k_error(logits, labels,k_=1):
    #logits1=tf.reshape(logits, [-1, 10])
    #labels1=tf.reshape(labels, [-1, 10])
    #print logits1,labels
    correct = tf.nn.in_top_k(logits, labels, k=k_)
    return tf.reduce_sum(tf.cast(correct, tf.float32))/subdivisions_batch_size


"""训练"""
def train_crack_captcha_cnn():   
    start_time = time.time()
    total_duration_time=0      # total_time

    with tf.Graph().as_default():  
        with tf.device('/cpu:0'):
            batch_x_train, batch_y_train = read_and_decode("train.tfrecords",40,phase="train")
            batch_x_test, batch_y_test = read_and_decode("test.tfrecords",32,phase="test")

        print "batch_x_train:",batch_x_train
        print "batch_y_train:",batch_y_train
        print "batch_x_test:",batch_x_test
        print "batch_y_test:",batch_y_test
    
        img_batch_train, label_batch_train = tf.train.shuffle_batch([batch_x_train, batch_y_train],
                                            batch_size=FLAGS.batch_size, capacity=int(FLAGS.train_size*0.02)+3*FLAGS.batch_size,
                                            min_after_dequeue=int(FLAGS.train_size*0.02),num_threads=4) 
        img_batch_test, label_batch_test = tf.train.batch([batch_x_test, batch_y_test],
                                            batch_size=FLAGS.batch_size, capacity=int(FLAGS.train_size*0.02)+3*FLAGS.batch_size,
                                            #min_after_dequeue=512,
                                            num_threads=4) 
        print "img_batch_train:",img_batch_train
        print "label_batch_train:",label_batch_train
        print "img_batch_test:",img_batch_test
        print "label_batch_test:",label_batch_test

        # Define 
        _labels = tf.placeholder(tf.int32,[None,])              # 标签
        _inputRGB = tf.placeholder(tf.float32,[None,ResNet.IMAGE_HEIGHT,ResNet.IMAGE_WIDTH,3])  # 图像
        global_ = tf.placeholder(tf.int32)                      # 迭代次数
        is_train = tf.placeholder(tf.bool)                      # train or test 用于BN层
        keep_prob = tf.placeholder(tf.float32)                  # dropout      
        
        with slim.arg_scope(ResNet.resnet_arg_scope(is_training=is_train)): 
            net, end_points = ResNet.resnet_20(_inputRGB, 10,is_training=is_train,keep_prob=keep_prob)
        
        loss=loss_fun(logits=net, labels=_labels)
        print "loss:",loss
        #loss_test=loss_fun(logits=logits_test, labels=label_batch_test)
        
        tf.summary.scalar('loss', loss)
        #tf.summary.scalar('loss_test', loss_test)        
        
        boundaries = [32000, 48000, 64000]
        values = [FLAGS.Initial_learning_rate, FLAGS.Initial_learning_rate/10.0, FLAGS.Initial_learning_rate/100.0,FLAGS.Initial_learning_rate/100.0]
        lr = tf.train.piecewise_constant(global_, boundaries, values)
             
        opt = tf.train.MomentumOptimizer(lr,momentum=0.9,use_nesterov=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


        grads = opt.compute_gradients(loss)
        #for grad, var in grads:
        #    print ("grad:",grad)
        #    print ("var:",var)

        # 删掉没梯度的参数, 倒序删除，减少麻烦
        for i in range(len(grads))[::-1]:
            if grads[i][0] is None:
                del grads[i]
        # 生成梯度缓存
        grads_cache = [tf.Variable(np.zeros(t[0].shape.as_list(), np.float32), trainable=False) for t in grads]
        # 清空梯度缓存op，每一 batch 开始前调用
        clear_grads_cache_op = tf.group(*[gc.assign(tf.zeros_like(gc)) for gc in grads_cache])
        # 累积梯度op，累积每个 sub batch 的梯度
        #print "zip(grads_cache, grads_vars):",zip(grads_cache, grads_vars)
        accumulate_grad_op = tf.group(*[gc.assign_add(gv[0]) for gc, gv in zip(grads_cache, grads)])
        # 求平均梯度，
        mean_grad = [gc/tf.to_float(subdivisions) for gc in grads_cache]
        # 组装梯度列表
        new_grads_vars = [(g, gv[1]) for g, gv in zip(mean_grad, grads)]

        apply_gradient_op = opt.apply_gradients(new_grads_vars)#, global_step=global_)
        #print "grads:",grads
        #print "apply_gradient_op:",apply_gradient_op
        #print "*update_ops:",update_ops
        
        train_op_new = tf.group(apply_gradient_op,*update_ops) 
       

        #train_op,lr = training(loss,global_step=global_)

        tf.summary.scalar('learning_rate', lr)
        accuracy = top_k_error(logits=end_points['predictions'] , labels=_labels, k_=1)
        
        tf.summary.scalar('accurate', accuracy) # display accurate in TensorBoard

        image_train = img_batch_train   
        image_test = img_batch_test   
        tf.summary.image("image_train", image_train)
        tf.summary.image("image_test", image_test)
        
        summary = tf.summary.merge_all()
        
        saver = tf.train.Saver(max_to_keep=0) 
        
        #sv = tf.train.Supervisor(logdir=FLAGS.log_dir)
        #sys.exit(0)
        
#------------------------------------------  吃光显存
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
##-------------------------------------------    
        
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
            #saver.restore(sess, "./logs/model.ckpt-50000")  #要改
            saver = tf.train.import_meta_graph('./logs/model.ckpt-50000.meta')
            saver.restore(sess, tf.train.latest_checkpoint("./logs/"))
            print "Model restored."  
            init_step=50000                               #要改
            for var in tf.trainable_variables():
                print var
            fine_tune_var_list = [v for v in tf.trainable_variables() ]




            #fine_tune_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
            #print "fine_tune_var_list:",fine_tune_var_list
            
            #load_weights( "./logs/model.ckpt-50000", sess )            
            
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        
        duration = time.time() - start_time
        total_duration_time+=duration
        start_time = time.time()
        max_acc=0
        max_acc_iter=0
        acc=0
        loss_sum = 0 
        for step in range(init_step,FLAGS.max_steps):
            loss_sum = 0 
            batch_x, batch_y= sess.run([img_batch_train, label_batch_train])
            if is_on_subdivisions:
                sess.run(clear_grads_cache_op) # 每一批开始前需要清空梯度缓存
                sub_loss_sum = 0
                for s in range(subdivisions):

                    x_sub_batch = batch_x[s * subdivisions_batch_size: (s + 1) * subdivisions_batch_size]
                    y_sub_batch = batch_y[s * subdivisions_batch_size: (s + 1) * subdivisions_batch_size]
            #        #_, learn_lr,loss_ = sess.run([train_op,lr, loss], feed_dict={ _inputRGB:x_sub_batch ,_labels:y_sub_batch,keep_prob: 0.8,is_train:True,global_:step})  
                    feed_dict = {_inputRGB: x_sub_batch, _labels: y_sub_batch,keep_prob: 0.5,is_train:True,global_:step}
                    _, los = sess.run([accumulate_grad_op, loss], feed_dict)
                    sub_loss_sum += los
                loss_sum += sub_loss_sum / subdivisions
                feed_dict = {_inputRGB: x_sub_batch, _labels: y_sub_batch,keep_prob: 0.5,is_train:True,global_:step}
                _ = sess.run([train_op_new,lr],feed_dict) # 梯度累积完成，开始应用梯度
                learn_lr = _[1]
                #print "_:",_

            #batch_x, batch_y= sess.run([img_batch_train, label_batch_train])

            #sys.exit(0)
            #_, learn_lr,loss_ = sess.run([train_op,lr, loss], feed_dict={ _inputRGB:batch_x ,_labels:batch_y,keep_prob: 0.8,is_train:True,global_:step})  
            #loss_ = sess.run(loss, feed_dict={ _inputRGB:batch_x ,_labels:batch_y,keep_prob: 0.8,is_train:True,global_:step})  
            #_, = sess.run([train_op_new], feed_dict={ _inputRGB:batch_x ,_labels:batch_y,keep_prob: 0.8,is_train:True,global_:step})  

            
            if (step+1) % FLAGS.log_frequency == 0:
                duration = time.time() - start_time
                total_duration_time+=duration
                start_time = time.time()
                
                #print "[%s] ljf-tf-train: Iter:%d/%d (%.1f examples/sec, %.3f sec/%d iters) ,loss=%.5f ,lr=%.5f"%(datetime.now(),(step+1),FLAGS.max_steps,FLAGS.batch_size*FLAGS.log_frequency/duration,duration,FLAGS.log_frequency,loss_,learn_lr)
                print "[%s] ljf-tf-train: Iter:%d/%d (%.1f examples/sec, %.3f sec/%d iters) ,loss=%.5f ,lr=%.5f"%(datetime.now(),(step+1),FLAGS.max_steps,FLAGS.batch_size*FLAGS.log_frequency/duration,duration,FLAGS.log_frequency,loss_sum,learn_lr)
            # 每100 step计算一次准确率  
            if (((step+1) % (FLAGS.log_frequency*5)== 0) or (step==0)): 
                acc_train=0
                acc_test=0
                for i in range(FLAGS.test_num):    #train
                    batch_x, batch_y= sess.run([img_batch_train, label_batch_train])    
                    sub_acc=0
                    for s in range(subdivisions):
                        x_sub_batch = batch_x[s * subdivisions_batch_size: (s + 1) * subdivisions_batch_size]
                        y_sub_batch = batch_y[s * subdivisions_batch_size: (s + 1) * subdivisions_batch_size]
                        sub_acc += sess.run(accuracy, feed_dict={_inputRGB:x_sub_batch ,_labels:y_sub_batch,keep_prob: 1.,is_train:False,global_:step})

                    acc_train+=sub_acc/subdivisions
                    #acc += sess.run(accuracy, feed_dict={_inputRGB:batch_x ,_labels:batch_y,keep_prob: 1.,is_train:False,global_:step})
                acc_train=acc_train/FLAGS.test_num
                           
                
                summary_str = sess.run(summary, feed_dict={_inputRGB:x_sub_batch ,_labels:y_sub_batch,is_train:False,keep_prob: 1.,global_:step})       # 训练的tensorboard
                summary_writer_train.add_summary(summary_str, step)
                summary_writer_train.flush()

                for i in range(FLAGS.test_num):    #test
                    batch_x, batch_y= sess.run([img_batch_test, label_batch_test]) 
                    sub_acc=0
                    for s in range(subdivisions):
                        x_sub_batch = batch_x[s * subdivisions_batch_size: (s + 1) * subdivisions_batch_size]
                        y_sub_batch = batch_y[s * subdivisions_batch_size: (s + 1) * subdivisions_batch_size]
                        sub_acc += sess.run(accuracy, feed_dict={_inputRGB:x_sub_batch ,_labels:y_sub_batch,keep_prob: 1.,is_train:False,global_:step})
                    #acc_test += sess.run(accuracy, feed_dict={ _inputRGB:batch_x ,_labels:batch_y,keep_prob: 1.,is_train:False,global_:step})
                    acc_test+=sub_acc/subdivisions
                acc_test=acc_test/FLAGS.test_num
                
                loss_test = sess.run(loss, feed_dict={ _inputRGB:x_sub_batch ,_labels:y_sub_batch,keep_prob: 1,is_train:True,global_:step})   # 测试的tensorboard
                
                summary_str = sess.run(summary, feed_dict={_inputRGB:x_sub_batch ,_labels:y_sub_batch,is_train:False,keep_prob: 1.,global_:step})
                summary_writer_test.add_summary(summary_str, step)
                summary_writer_test.flush()                            

                duration = time.time() - start_time
                total_duration_time+=duration
                start_time = time.time()
                
                if(acc_test>max_acc):
                    max_acc=acc_test
                    max_acc_iter=step+1
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step+1)  

                print "[%s] ljf-tf-test : train accuracy:%.4f%%,  test accuracy:%.4f%% ,  max accuracy:%.4f%% , max accuracy step:%d"%(datetime.now(),acc_train*100,acc_test*100,max_acc*100,max_acc_iter)       
                print "[%s] ljf-tf-test : test  loss=%.5f,       test time:%.3f sec,     total time:%.3f sec"%(datetime.now(),loss_test,duration,total_duration_time)

            if (step+1) % 10000 == 0 or (step+1)==FLAGS.max_steps: 
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step+1)  

            
        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()   





def main(argv=None):  # pylint: disable=unused-argument
  print("Tensorflow version " + tf.__version__)
  if clear==True:
      if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
      tf.gfile.MakeDirs(FLAGS.log_dir)
  train_crack_captcha_cnn()                                            


if __name__ == '__main__':
  tf.app.run()
