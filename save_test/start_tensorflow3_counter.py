# -*- coding: utf-8 -*-
"""
Saver 
Please read readme.txt
"""

# 进入一个交互式 TensorFlow 会话.
import tensorflow as tf

if __name__ == '__main__':  

    state = tf.Variable(0, name="counter")

    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    

    init_op = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    
    # 启动图, 运行 op
    #with tf.Session() as sess:
    sess = tf.Session()
    # 运行 'init' op
    
    #sess.run(init_op)
    saver.restore(sess, "counter_model/counter_model.ckpt")
    print "Model restored."
    #    # 打印 'state' 的初始值
    #print sess.run(state)
    #print state.eval()
    
    print ' '
    # 运行 op, 更新 'state', 并打印 'state'
    for _ in range(3):
        sess.run(update)
        print sess.run(state)
        #print state.eval()
    save_path = saver.save(sess, "counter_model/counter_model.ckpt")
    print "Model saved in file: ", save_path
    # 输出:







