演示了Saver的功能，可以掉电保存，停了继续，只要保存了模型


..............>python start_tensorflow3_counter.py    执行即可（必须是.py当前目录下）

以i计数器为例

开saver.restore(sess, "counter_model/counter_model.ckpt")这条，屏蔽sess.run(init_op)这条，是继续
反之，是从头计数
