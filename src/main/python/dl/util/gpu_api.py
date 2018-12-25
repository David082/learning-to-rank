# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/23
version : 
"""
# ------ nvidia-smi
"""
nvidia-smi
nvidia-smi -a

Linux下实时查看GPU状态
https://blog.csdn.net/weixin_40241703/article/details/81111478
最常用的参数是 -n， 后面指定是每多少秒来执行一次命令。
监视显存：假设我们设置为每 30s 显示一次显存的情况：
watch -n 30 nvidia-smi
"""

# ------ keras指定运行时显卡及限制GPU用量
# https://blog.csdn.net/github_36326955/article/details/79910448
# https://blog.csdn.net/A632189007/article/details/77978058
"""
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 进行配置，每个GPU使用60%上限现存
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)
"""

# ------ nvidia-smi 命令解读
# https://blog.csdn.net/sallyxyl1993/article/details/62220424
"""
第一栏的Fan：N/A是风扇转速，从0到100%之间变动，这个速度是计算机期望的风扇转速，实际情况下如果风扇堵转，可能打不到显示的转速。
有的设备不会返回转速，因为它不依赖风扇冷却而是通过其他外设保持低温（比如我们实验室的服务器是常年放在空调房间里的）。
第二栏的Temp：是温度，单位摄氏度。
第三栏的Perf：是性能状态，从P0到P12，P0表示最大性能，P12表示状态最小性能。
第四栏下方的Pwr：是能耗，上方的Persistence-M：是持续模式的状态，持续模式虽然耗能大，但是在新的GPU应用启动时，花费的时间更少，这里显示的是off的状态。
第五栏的Bus-Id是涉及GPU总线的东西，domain:bus:device.function
第六栏的Disp.A是Display Active，表示GPU的显示是否初始化。
第五第六栏下方的Memory Usage是显存使用率。
第七栏是浮动的GPU利用率。
第八栏上方是关于ECC的东西。
第八栏下方Compute M是计算模式。
下面一张表示每个进程占用的显存使用率。
"""

# ------ 检测tensorflow是否使用gpu进行计算
# https://blog.csdn.net/castle_cc/article/details/78389082
"""
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
"""

# ------ cuda runtime error(2): out of memory
"""
pytorch 减小显存消耗，优化显存使用，避免out of memory
https://blog.csdn.net/qq_28660035/article/details/80688427
显存跟踪
可以实时地打印我们使用的显存以及哪些Tensor使用了我们的显存
https://github.com/Oldpan/Pytorch-Memory-Utils

pytorch模型提示超出内存cuda runtime error(2): out of memory
https://ptorch.com/news/160.html
"""

# ------ 计算模型参数大小
# http://www.aibbt.com/a/44242.html
"""
import numpy asnp
# 模型内参数个数
para =sum([np.prod(list(p.size())) for p in model.parameters()])
# float32 占 4个 Byte
type_size =4
model_size =para * type_size /1024/1024
print(model_size)
# 约为0.43M
"""

# ------ tensorflow-gpu测试代码
# https://blog.csdn.net/william_hehe/article/details/79615894
"""
import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')
    b = tf.constant([1.0,2.0,3.0],shape=[3],name='b')
with tf.device('/gpu:1'):
    c = a+b

#注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
#因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
print(sess.run(c))
"""

# ------ pytorch-gpu测试代码
# https://blog.csdn.net/u013063099/article/details/79246984
"""
import torch
x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)
"""
# ------ 清理GPU显存
# https://blog.csdn.net/a694262054/article/details/80020150
# sudo fuser -v /dev/nvidia* #查找占用GPU资源的PID
# nvidia-smi -L # list all available NVIDIA devices
