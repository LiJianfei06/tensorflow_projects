python tensorflow_captcha_generate_samples.py   生成样本，必须在当前目录下,默认100k个训练样本，10K个测试集，1000个用来测试预测的(跑网络时用不到)

read_write_data_alphabet_n.py   		生成train.tfrecords和val.tfrecords训练文件(后面训练我是用16线程读取的，视自己计算机性能而定)

python tensorflow_train_mynet.py   		训练，必须在当前目录下
...........................>tensorboard --logdir="logs"     查看训练曲线，必须在当前目录下

mynet.py   					网络结构什么的都在这里面
python eval_captcha.py   			预测，必须在当前目录下



支持tfrecord   		 2017-10-19
支持saver 断电续训练   	 2017-10-22
