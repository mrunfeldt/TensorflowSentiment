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
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  sess = tf.InteractiveSession()
  fresh_run = FLAGS.continue_from == -1
  if fresh_run:
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784], name = "x")
    W = tf.Variable(tf.zeros([784, 10]), name = "W")
    b = tf.Variable(tf.zeros([10]), name = "b")
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10], "y_")

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_", y_)
    tf.add_to_collection("W", W)
    tf.add_to_collection("b", b)
  else:
    model = 'my-model-{}'.format(FLAGS.continue_from)
    new_saver = tf.train.import_meta_graph('{}.meta'.format(model))
    new_saver.restore(sess, './{}'.format(model))
    x = tf.get_collection("x")[0]
    W = tf.get_collection("W")[0]
    b = tf.get_collection("b")[0]
    y = tf.get_collection("y")[0]
    y_ = tf.get_collection("y_")[0]
    
    
  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  
  # Train
  if fresh_run:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    
  for i in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if fresh_run:
      saver.save(sess, 'my-model', global_step = i)
    
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  parser.add_argument('--continue-from', type=int, default=-1)
  FLAGS = parser.parse_args()
  tf.app.run()
