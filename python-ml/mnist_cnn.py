#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

#初始化权重向量
def weight_variable(shape):
    #shape:[5, 5, 1, 32] 一维张量，5,5是patch大小， 1：是输入管道？ 32：输出通道数目， 32维特征
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)

#初始化偏置
def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)

#卷积:步长(stride_size=[1,1,1,1]), 边距: (padding_size=)
def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2d(x):
   return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')


#see also https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py
def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])

  #第一层卷积
  ##讲x变成4d向量， 其中2，3维对应图片的宽、高， 最后一维表示颜色
  ##灰度图是1，rgb彩色图，则是3
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  ##权重初始化
  W_conv1 = weight_variable([5, 5, 1, 32])
  ##偏置初始化
  b_conv1 = bias_variable([32])
  ##卷积、编制、整形
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  ##max pooling池化                     
  h_pool1 = max_pool_2d(h_conv1)
  
  #第二层卷积
  ##权重初始化
  W_conv2 = weight_variable([5, 5, 32, 64])
  ##偏置初始化
  b_conv2 = bias_variable([64])
  ##卷积、编制、整形
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  ##max pooling池化                     
  h_pool2 = max_pool_2d(h_conv2)

  #密集连接层:???TODO
  W_fc1 = weight_variable([7*7*64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
  #dropout: 减少过拟合
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  #输出层
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  
  #训练和评估模型
  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
  #存储训练模型
  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

 
  with tf.Session() as sess:
    #初始化  
   sess.run(tf.global_variables_initializer())  
   summary = tf.summary.merge_all()
   summary_writer = tf.summary.FileWriter("/Users/didi/workspace/data/tensorflow/tmp/test_logs",  sess.graph)
    
   #训练
   for i in range(10):
      batch = mnist.train.next_batch(50)
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      if i % 10 == 0:
          summary_str, acc = sess.run([summary, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
          summary_writer.add_summary(summary_str, i)
          print('Accuracy at step %s: %s' % (i, acc))
    #print('test accuracy %g' % accuracy.eval(feed_dict={
        #x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
     #Visualize the status
   summary_writer.close() 
  
  
  #sess = tf.Session()
  #init = tf.global_variables_initializer()
  #sess.run(init)
  
  #for i in range(20000):  #(0,1000) 迭代次数
    #batch = mnist.train.next_batch(50)  #每次50个样本
    #x = batch[0]
    #y_ = batch[1]
    #keep_prob = 1.0
    #if i % 100 == 0:
        #train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        #print("step %d, training accuracy %g"%(i, train_accuracy))
    #train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob:0.5})    
    
  #print ('test accuracy %g'%accuracy.eval(feed_dict={
          #x: mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0})
         #)  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/Users/didi/workspace/data/tensorflow/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
