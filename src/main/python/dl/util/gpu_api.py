# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/23
version : 
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
