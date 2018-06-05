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

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model 
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  #运算 http://www.tensorfly.cn/tfdoc/api_docs/python/math_ops.html
  y = tf.matmul(x, W) + b #矩阵乘法

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  # http://www.tensorfly.cn/tfdoc/api_docs/python/nn.html
  
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  
  #返回boolean数组,eg[1,0,1,1]
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  #计算平均值
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  with tf.Session() as sess:
     #添加可视化
     merged_summary = tf.summary.merge_all()
     train_writer = tf.summary.FileWriter("/Users/didi/workspace/data/tensorflow/tmp/train_logs", sess.graph)
     #test_writer = tf.summary.FileWriter("/Users/didi/workspace/data/tensorflow/tmp/test_logs", sess.graph)

     sess.run(tf.global_variables_initializer())
     total_step = 0
     for _ in range(10):  #(0,1000) 迭代次数c
        total_step += 1
        batch = mnist.train.next_batch(100)  #每次50个样本
        summary_str, train_str = sess.run([merged_summary,train_step], feed_dict={x: batch[0], y_: batch[1]})
        train_writer.add_summary(summary_str, total_step) 
        #sess.run(train_step, feed_dict)
       
        #if total_step % 10 == 0 : 
            #summary_str, acc = sess.run([merged_summary, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) 
            #print(summary_str)
            #train_writer.add_summary(summary_str, total_step)  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/Users/didi/workspace/data/tensorflow/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


