# coding:utf-8

import tensorflow as tf
import numpy as np

# 输出y -> Softmax -> y_交叉熵
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=y, labels=tf.argmax(y_, 1))
cem = tf.reduce_mean(ce)

# 学习率衰减
# staircase=True 学习率阶梯衰减
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE, global_step, LEARNING_EATE_STEP, LEARNING_RATE_DECAY, staircase=True)

# 滑动平均（影子） W,b
global_step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
ema_op = ema.apply(tf.trainable_variables())

# 正则化
# regularizer 正则化权重
tf.add_to_collection(
    'losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
loss += tf.add_n(tf.get_collection('losses'))

# 卷积
tf.nn.conv2d([batch, 5, 5, 1],  # 输入描述：分辨率5x5，通道数为1
             [3, 3, 1, 16],  # 卷积核描述：核大小3x3，通道数为1，核个数为16
             [1, 1, 1, 1],  # 中间1，1：行步长为1，列步长为1，其余1固定
             padding='VALID')  # 'SAME'

# 池化
# 最大值池化：提取图片纹理
# 均值池化：保留背景特征
tf.nn.max_pool([batch, 28, 28, 6],
               [1, 2, 2, 1],  # 行列大小为2
               [1, 2, 2, 1],
               padding='SAME')
# tf.nn.avg_pool

# Dropout
# tf.nn.dropout()