#coding:utf-8

import tensorflow as tf
import numpy as np

# 输出y -> Softmax -> y_交叉熵
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cem = tf.reduce_mean(ce)

# 学习率衰减
# staircase=True 学习率阶梯衰减
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_EATE_STEP, LEARNING_RATE_DECAY, staircase=True)

# 滑动平均（影子） W,b
global_step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
ema_op = ema.apply(tf.trainable_variables())

# 正则化 
# regularizer 正则化权重
tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
loss += tf.add_n(tf.get_collection('losses'))
