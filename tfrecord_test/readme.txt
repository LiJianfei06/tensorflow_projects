read_write_data_test.py 演示了tfrecord的功能,把read_test_alphabet1文件夹下的文件按照0,1,2……的顺序打标签，测试打乱取batch和顺序取batch
..............>python read_write_data_test.py    执行即可（必须是.py当前目录下），会看到输出的两大段


read_write_data_alphabet.py 演示了tfrecord的功能,把read_test_alphabet1文件夹下的文件按照0-62打标签(数字加字母大小写)，测试打乱取batch
..............>python read_write_data_alphabet.py    执行即可（必须是.py当前目录下），会看到输出的两大段


read_write_data_alphabet_n.py 演示了tfrecord的功能,把read_test_alphabet4文件夹下的文件按照0-62打标签
(最后两位代表第一位字符，倒着来方便n位验证码)，测试打乱取batch和顺序取batch
..............>python read_write_data_alphabet_n.py    执行即可（必须是.py当前目录下），会看到输出的两大段
